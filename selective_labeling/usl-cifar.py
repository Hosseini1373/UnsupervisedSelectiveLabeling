# %%
import os
os.environ["USL_MODE"] = "USL"

import numpy as np
import torch
import models.resnet_cifar_cld as resnet_cifar_cld
import utils
from utils import cfg, logger, print_b

utils.init(default_config_file="configs/cifar10_usl.yaml")

logger.info(cfg)

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)

model = resnet_cifar_cld.__dict__[cfg.MODEL.ARCH](
    low_dim=128, pool_len=4, normlinear=True).cuda()
model.load_state_dict(utils.single_model(checkpoint["model"]))
model.eval()

logger.info("model: {}".format(model))


# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "cifar10", "cifar100"], f"{cfg.DATASET.NAME} is not cifar10 or cifar100"
cifar100 = cfg.DATASET.NAME == "cifar100"
num_classes = 100 if cifar100 else 10

train_memory_dataset, train_memory_loader = utils.train_memory_cifar(
    root_dir=cfg.DATASET.ROOT_DIR,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100)

targets = torch.tensor(train_memory_dataset.targets)



#Logging some samples of data:

logger.info("targets.shape: {}".format(targets.shape))
# Fetch one batch of images and their labels
for images, labels in train_memory_loader:
    break  # This breaks the loop, so we only get the first batch

# Convert the images and labels to a human-readable format
# Assuming images are tensors and labels are either tensors or lists
images_np = images.numpy()  # Convert images to a NumPy array if they're tensors
labels_np = labels.numpy()  # Convert labels to a NumPy array if they're tensors

# Log the shapes of images and labels for understanding
logger.info(f"Batch of images shape: {images_np.shape}")
logger.info(f"Batch of labels shape: {labels_np.shape}")

# Optionally, log or print the first few labels to understand the dataset better
num_samples_to_display = 5  # Adjust this to display more or fewer samples
for i in range(num_samples_to_display):
    logger.info(f"Sample {i} label: {labels_np[i]}")
    # If you want to display images, consider converting them to a suitable format
    # This part is tricky in a logging context because images need to be visualized


# %%
print_b("Loading feat list")
feats_list = utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

logger.info("feats_list: {}".format(feats_list))


# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
logger.info("d_knns: {} ind_knns: {}".format(d_knns,ind_knns))
score_first_order = 1/neighbors_dist
logger.info("score_first_order: {}".format(score_first_order))

# %%

logger.info("cfg.USL.NUM_SELECTED_SAMPLES: {}".format(cfg.USL.NUM_SELECTED_SAMPLES))
num_centroids, final_sample_num = utils.get_sample_info_cifar(
    chosen_sample_num=cfg.USL.NUM_SELECTED_SAMPLES)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))

# %%



recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
for kMeans_seed in cfg.USL.SEEDS:
    print_b(f"Running k-Means with seed {kMeans_seed}")
    if final_sample_num <= 40:
        # This is for better reproducibility, but has low memory usage efficiency.
        force_no_lazy_tensor = True
    else:
        force_no_lazy_tensor = False
    logger.info(f"force_no_lazy_tensor is set to {'True' if force_no_lazy_tensor else 'False'} for better reproducibility.")

    # This has side-effect: it calls torch.manual_seed to ensure the seed in k-Means is set.
    # Note: NaN in centroids happens when there is no corresponding sample which belongs to the centroid
    cluster_labels, centroids = utils.run_kMeans(feats_list, num_centroids, final_sample_num, Niter=cfg.USL.K_MEANS_NITERS,
                                                 recompute=recompute_num_dependent, seed=kMeans_seed, force_no_lazy_tensor=force_no_lazy_tensor)

    logger.info(f"k-Means completed with {num_centroids} centroids for seed {kMeans_seed}.")
    if np.isnan(centroids).any():
        logger.warning("NaN values found in centroids.")
                    
    # In the context of your code, after performing k-Means clustering, a 
    # regularization-based selection process is applied. Regularization is a 
    # technique used to prevent overfitting by introducing additional information (
    # or constraints) to the optimization problem. In this case, regularization 
    # likely influences the selection of samples based on additional criteria 
    # beyond their proximity to cluster centroids, such as diversity or representation 
    # across different clusters.

    print_b("Getting selections with regularization")
    selected_inds = utils.get_selection(utils.get_selection_with_reg, feats_list, neighbors_dist, cluster_labels, num_centroids, final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS, w=cfg.USL.REG.W,
                                        momentum=cfg.USL.REG.MOMENTUM, horizon_dist=cfg.USL.REG.HORIZON_DIST, alpha=cfg.USL.REG.ALPHA, verbose=True, seed=kMeans_seed, recompute=recompute_num_dependent, save=True)


    counts = np.bincount(np.array(train_memory_dataset.targets)[selected_inds])
    logger.info(f"Class counts: {sum(counts > 0)}")
    logger.info(f"Counts per class: {counts.tolist()}")
    logger.info(f"Max count: {counts.max()}, Min count: {counts.min()}")
    logger.info(f"Number of selected indices: {len(selected_inds)}")
    logger.debug(f"Selected indices: {repr(selected_inds)}")
    
    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))
