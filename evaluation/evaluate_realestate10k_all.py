# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse

from evaluate_perceptualsim import \
    compute_perceptual_similarity as compute_both
from evaluate_perceptualsim_centercrop_invis import \
    compute_perceptual_similarity as compute_invis
from evaluate_perceptualsim_centercrop_vis import \
    compute_perceptual_similarity as compute_vis


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, default="")
    args.add_argument("--pred_image", type=str, default="output_image_.png")
    args.add_argument("--target_image", type=str, default="tgt_image_.png")
    args.add_argument("--take_every_other", action="store_true", default=False)
    args.add_argument("--output_file", type=str, default="eval_out")

    opts = args.parse_args()

    folder = opts.folder
    if not folder.endswith('/'):
        folder = folder + '/'
    pred_img = opts.pred_image
    tgt_img = opts.target_image

    results_both = compute_both(
        folder, pred_img, tgt_img, opts.take_every_other
    )
    results_both = {('Both/' + k): v for k, v in results_both.items()}

    results_invis = compute_invis(
        folder, pred_img, tgt_img, opts.take_every_other
    )
    results_invis = {('InVis/' + k): v for k, v in results_invis.items()}

    results_vis = compute_vis(
        folder, pred_img, tgt_img, opts.take_every_other
    )
    results_vis = {('Vis/' + k): v for k, v in results_vis.items()}

    results = {}
    results.update(results_both)
    results.update(results_invis)
    results.update(results_vis)

    key_print_list = [
        "Both/PSNR",
        "InVis/PSNR",
        "Vis/PSNR",
        "Both/SSIM",
        "InVis/SSIM",
        "Vis/SSIM",
        "Both/Perceptual similarity",
        "InVis/Perceptual similarity",
        "Vis/Perceptual similarity",
    ]

    val_print_list = []
    for key in key_print_list:
        print("%s for %s: \n" % (key, opts.folder))
        print(
            "\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1])
        )
        val_print_list.append(f"{results[key][0]:0.4f}")

    print('\n')
    print('-' * 80)
    print('copy-paste metrics:')
    print(','.join(key_print_list))
    print(','.join(val_print_list))
    print('-' * 80)
    print('\n')

    output_file = opts.output_file + (
        '.every_other_ON' if opts.take_every_other else '.every_other_OFF'
    )
    f = open(output_file, 'w')
    for key in key_print_list:
        f.write("%s for %s: \n" % (key, opts.folder))
        f.write(
            "\t {:0.4f} | {:0.4f} \n".format(results[key][0], results[key][1])
        )

    f.write(','.join(key_print_list))
    f.write('\n')
    f.write(','.join(val_print_list))
    f.write('\n')

    f.close()
