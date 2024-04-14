import os
import torch
from torch import nn, optim
from typing import Optional
import random
from pathlib import Path
from derang.diacritizer import TorchDiacritizer
from derang.dataset import load_validation_data, load_test_data
import derang.utils as utls
from derang.logger import _LOGGER
from lightning.pytorch import LightningModule
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from more_itertools import take
from x_transformers.x_transformers import Encoder

class HiGRU(nn.Module):
    def __init__(self, *args, pad_idx, sum_bidirection=False, **kwargs):
        super().__init__()
        self.gru = nn.GRU(*args, **kwargs)
        self.batch_first = self.gru.batch_first
        self.hidden_size = self.gru.hidden_size
        self.bidirectional = self.gru.bidirectional
        self.pad_idx = pad_idx
        self.sum_bidirectionnal = sum_bidirection
        
    def forward(self, input: torch.Tensor, lengths: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        packed_input = pack_padded_sequence(input, lengths, batch_first=self.batch_first, enforce_sorted=self.training)
        output, hx = self.gru(packed_input, hx)
        output, _lengths = pad_packed_sequence(output, batch_first=self.batch_first, padding_value=self.pad_idx)
        if self.bidirectional and self.sum_bidirectionnal:
            output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
        return output.tanh()
    
    
class AlefModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        # @todo: first need to introduce config.len_target_symbols etc.
        hparams  = {
            "inp_vocab_size": config.len_input_symbols,
            "targ_vocab_size": config.len_target_symbols,
            "input_pad_idx": config.text_encoder.input_pad_id,
            "target_pad_idx": config.text_encoder.target_pad_id,
            **config.config
        }
        
        # inherig from ligthning module
        self.save_hyperparameters(hparams)
        
        self.config = config
        self.training_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}
        
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.hparams.target_pad_idx)
        self._build_layers(
            d_model=self.hparams.d_model,
            inp_vocab_size=self.hparams.inp_vocab_size,
            targ_vocab_size=self.hparams.targ_vocab_size,
            input_pad_idx=self.hparams.input_pad_idx,
            target_pad_idx=self.hparams.target_pad_idx
        )
        
    def _build_layers(self, d_model, inp_vocab_size, targ_vocab_size, input_pad_idx, target_pad_idx):
        self.emb = nn.Embedding(inp_vocab_size, d_model, padding_idx = input_pad_idx)
        nn.init.normal_(self.emb.weight, -1, 1)
        self.gru = HiGRU(d_model, d_model,
                         num_layers=6, 
                         batch_first=True, 
                         bidirectional=True,
                         dropout=0.1, 
                         pad_idx=input_pad_idx, 
                         sum_bidirection=True
        )
        self.gru_dropout=nn.Dropout(0.2)
        self.attn_layers = Encoder(
            dim=d_model,
            depth=6,
            heads=8,
            layer_dropout=0.2,
            ff_dropout=0.2,
            ff_relu_squared=True,
            rel_pos_bias=True,
            onnxable=True
        )
        
        self.res_layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, targ_vocab_size)
        
    
    def forward(self, inputs, lengths):
        X = inputs.to(self.device)
        lengths = lengths.to('cpu')
        length_mask = utls.sequence_mask(lengths, X.size(1)).type_as(X)
        
        X = self.emb(X)
        
        gru_out = self.gru(X, lengths)
        gru_out = self.gru_dropout(gru_out)
        
        attn_mask = length_mask.bool().logical_not()
        attn_out = self.attn_layers(X, mask=attn_mask)
        
        # best weighting factors for inference: gru: 9, enc: 5, attn: 8
        X = (gru_out * 8) + (attn_out * 10)
        X = nn.functional.leaky_relu(X)
        X = self.res_layernorm(X)
        
        X = self.fc_out(X)
        return X
    
    
    def predict(self, inputs, lengths):
        output = self(inputs, lengths)
        logits = output.softmax(dim=2)
        predictions = torch.argmax(logits, dim=2)
        return predictions.byte(), logits
    
    
    def _process_batch(self, batch):
        predictions = self(batch["src"], batch["lengths"])
        target = batch["target"].contiguous()
        
        predictions = predictions.view(-1, predictions.shape[-1])
        targets = batch["target"].view(-1)
        
        diac_loss = self.criterion(predictions.to(self.device), targets.to(self.device))
        diac_accuracy = utls.categorical_accuracy(predictions.to(self.device), targets.to(self.device), self.hparams.target_pad_idx, device=self.device)
        return {"loss": diac_loss, "accuracy": diac_accuracy}    
    
    
    def training_step(self, batch, batch_idx):
        metrics = self._process_batch(batch)
        for name, val in metrics.items():
            self.training_step_outputs.setdefault(name, []).append(val)
            self.log(name, val)
        return metrics["loss"]
    
    
    def validation_step(self, batch, batch_idx):
        metrics = self._process_batch(batch)
        for name, val in metrics.items():
            self.val_step_outputs.setdefault(f"val_{name}", []).append(val)
            self.log(f"val_{name}", val)
        
    def test_step(self, batch, batch_idx):
        metrics = self._process_batch(batch)
        for name, val in metrics.items():
            self.test_step_outputs.setdefault(f"test_{name}", []).append(val)
            self.log(f"test_{name}", val)
        return metrics
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas = tuple(self.hparams.adam_betas), weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.hparams.lr_factor, patience=self.hparams.lr_patience, min_lr=self.hparams.min_lr, mode='min', cooldown=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    
    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.training_step_outputs)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_step_outputs)
        if ((self.current_epoch + 1) % self.hparams.evaluate_with_error_rates_epochs) == 0:
            data_loader = load_validation_data(self.config)
            diacritizer = TorchDiacritizer(self.config, model=self)
            error_rates = self.evaluate_with_error_rates(
                diacritizer,
                data_loader=data_loader,
                num_batches=self.config.config["error_rates_n_batches"],
                predictions_dir=Path(self.trainer.log_dir).joinpath("predictions"),
            )
            self.log_dict({
                k.replace("*", "_star"): v
                for (k, v) in error_rates.items()
            })
            _LOGGER.info("Error Rates:\n" + utls.format_error_rates_as_table(error_rates))


    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_step_outputs)
        data_loader = load_test_data(self.config)
        diacritizer = TorchDiacritizer(self.config, model=self)
        error_rates = self.evaluate_with_error_rates(
            diacritizer,
            data_loader=data_loader,
            num_batches=self.config["error_rates_n_batches"],
            predictions_dir=Path(self.trainer.log_dir).joinpath("predictions"),
        )
        _LOGGER.info("Error Rates:\n" + utls.format_error_rates_as_table(error_rates))


    def _log_epoch_metrics(self, metrics):
        for name, values in metrics.items():
            epoch_metric_mean = torch.stack(values).mean()
            self.log(name, epoch_metric_mean)
            values.clear()
            
    
    @classmethod
    def evaluate_with_error_rates(cls, diacritizer, data_loader, num_batches, predictions_dir):
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        all_orig = []
        all_predicted = []
        results = {}
        num_batches_to_take = min(num_batches, len(data_loader))
        for batch in tqdm(take(num_batches_to_take, data_loader), total=num_batches_to_take, desc="predicting", unit="batch"):
            gt_lines = batch["original"]
            predicted, __ = diacritizer.diacritize_text(gt_lines)
            all_orig.extend(gt_lines)
            all_predicted.extend(predicted)
        
        orig_path = os.path.join(predictions_dir, f"original.txt")
        with open(orig_path, "w", encoding="utf-8") as file:
            lines = "\n".join(sent for sent in all_orig)
            file.write(lines)
            
        predicted_path = os.path.join(predictions_dir, f"predicted.txt")
        with open(predicted_path, "w", encoding="utf8") as file:
            lines = "\n".join(sent for sent in all_predicted)
            file.write(lines)
        
        
        try:
            results = utls.calculate_error_rates(orig_path, predicted_path)
        except:
            _LOGGER.error("Failed to calculate DER/WER statistics", exc_info=True)
            results = {"DER": 100.0, "WER": 100.0, "DER*": 100.0, "WER*": 100.0}

            
        return results

