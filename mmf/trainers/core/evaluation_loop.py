# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import is_master


logger = logging.getLogger(__name__)


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):
                report = self._forward(batch)
                self.update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            combined_report.metrics = self.metrics(combined_report, combined_report)
            self.update_meter(combined_report, meter, eval_mode=True)

            # enable train mode again
            self.model.train()

        _print_psnr_ssim_perc_sim(meter)
        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, torch.device("cuda"))
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            logger.info("Finished predicting")
            self.model.train()


def _print_psnr_ssim_perc_sim(meter):
    meters = meter.meters
    keys_to_print = [
        "PSNR",
        "PSNR_InVis",
        "PSNR_Vis",
        "SSIM",
        "SSIM_InVis",
        "SSIM_Vis",
        "PercSim",
        "PercSim_InVis",
        "PercSim_Vis",
    ]
    key_print_list = []
    val_print_list = []
    for k_print in keys_to_print:
        for k, v in meters.items():
            if k.endswith(k_print):
                key_print_list.append(k)
                val_print_list.append(f"{v.global_avg:.4f}")

    if len(key_print_list) > 0:
        print('\n')
        print('-' * 80)
        print('copy-paste metrics:')
        print(','.join(key_print_list))
        print(','.join(val_print_list))
        print('-' * 80)
        print('\n')
