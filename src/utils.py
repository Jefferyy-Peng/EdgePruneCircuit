import pickle

import torch
from torchvision import datasets
from PIL import Image

class SubsetFolder:
    def __init__(self, dataset, class_id):
        if isinstance(class_id, int):
            samples = np.array(dataset.samples, )[np.array(dataset.targets) == class_id]
        elif isinstance(class_id, list):
            class_id = np.array(class_id)
            mask = np.isin(np.array(dataset.targets), class_id)
            samples = np.array(dataset.samples, )[mask]
        self.samples = [(row[0], int(row[1])) for row in samples.tolist()]

    def __len__(self):
        return len(self.samples)

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, processor=None, transform=None, select_class=None):
        dataset = datasets.ImageFolder(root=root_dir)
        self.processor = processor
        self.transform = transform
        if select_class is not None:
            subset_dataset = SubsetFolder(dataset, select_class)
            self.dataset = subset_dataset
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")  # Use processor
            return inputs["pixel_values"].squeeze(0), label
        if self.transform is not None:
            inputs = self.transform(image)
            return inputs, label


# CIFAR-10.1 dataset, by Rebecca Roelofs and Ludwig Schmidt
# Copying the utils from there for convenience.

import os
import pathlib
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as datasets
import numpy as np

# Map from ImageNet renditions indices to ImageNet indices.
r_indices = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105,
             107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187,
             195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267,
             269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327,
             330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393,
             397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476,
             483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658,
             701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883,
             889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
             967, 980, 981, 983, 988]

# Map from ImageNet-A indices to ImageNet indices.
a_indices = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105,
             107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295,
             298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336,
             347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438,
             445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552,
             557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668,
             677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786,
             792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880,
             888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972,
             980, 981, 984, 986, 987, 988]

v2_indices = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117,
              118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136,
              137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 155,
              156, 157, 158, 159, 16, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173, 174,
              175, 176, 177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 19, 190, 191, 192, 193,
              194, 195, 196, 197, 198, 199, 2, 20, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 21, 210, 211, 212,
              213, 214, 215, 216, 217, 218, 219, 22, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 23, 230, 231,
              232, 233, 234, 235, 236, 237, 238, 239, 24, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 25, 250,
              251, 252, 253, 254, 255, 256, 257, 258, 259, 26, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 27,
              270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 28, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
              29, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 3, 30, 300, 301, 302, 303, 304, 305, 306, 307, 308,
              309, 31, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 32, 320, 321, 322, 323, 324, 325, 326, 327,
              328, 329, 33, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 34, 340, 341, 342, 343, 344, 345, 346,
              347, 348, 349, 35, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 36, 360, 361, 362, 363, 364, 365,
              366, 367, 368, 369, 37, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 38, 380, 381, 382, 383, 384,
              385, 386, 387, 388, 389, 39, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 4, 40, 400, 401, 402, 403,
              404, 405, 406, 407, 408, 409, 41, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 42, 420, 421, 422,
              423, 424, 425, 426, 427, 428, 429, 43, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 44, 440, 441,
              442, 443, 444, 445, 446, 447, 448, 449, 45, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 46, 460,
              461, 462, 463, 464, 465, 466, 467, 468, 469, 47, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 48,
              480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 49, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 5,
              50, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 51, 510, 511, 512, 513, 514, 515, 516, 517, 518,
              519, 52, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 53, 530, 531, 532, 533, 534, 535, 536, 537,
              538, 539, 54, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 55, 550, 551, 552, 553, 554, 555, 556,
              557, 558, 559, 56, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 57, 570, 571, 572, 573, 574, 575,
              576, 577, 578, 579, 58, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 59, 590, 591, 592, 593, 594,
              595, 596, 597, 598, 599, 6, 60, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 61, 610, 611, 612, 613,
              614, 615, 616, 617, 618, 619, 62, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 63, 630, 631, 632,
              633, 634, 635, 636, 637, 638, 639, 64, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 65, 650, 651,
              652, 653, 654, 655, 656, 657, 658, 659, 66, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 67, 670,
              671, 672, 673, 674, 675, 676, 677, 678, 679, 68, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 69,
              690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 7, 70, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709,
              71, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 72, 720, 721, 722, 723, 724, 725, 726, 727, 728,
              729, 73, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 74, 740, 741, 742, 743, 744, 745, 746, 747,
              748, 749, 75, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 76, 760, 761, 762, 763, 764, 765, 766,
              767, 768, 769, 77, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 78, 780, 781, 782, 783, 784, 785,
              786, 787, 788, 789, 79, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 8, 80, 800, 801, 802, 803, 804,
              805, 806, 807, 808, 809, 81, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 82, 820, 821, 822, 823,
              824, 825, 826, 827, 828, 829, 83, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 84, 840, 841, 842,
              843, 844, 845, 846, 847, 848, 849, 85, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 86, 860, 861,
              862, 863, 864, 865, 866, 867, 868, 869, 87, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 88, 880,
              881, 882, 883, 884, 885, 886, 887, 888, 889, 89, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 9, 90,
              900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 91, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919,
              92, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 93, 930, 931, 932, 933, 934, 935, 936, 937, 938,
              939, 94, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 95, 950, 951, 952, 953, 954, 955, 956, 957,
              958, 959, 96, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 97, 970, 971, 972, 973, 974, 975, 976,
              977, 978, 979, 98, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 99, 990, 991, 992, 993, 994, 995,
              996, 997, 998, 999]


class ImageNet(Dataset):

    def __init__(self, root, split='train', num_examples=None, transform=None, processor=None, seed=0):
        super().__init__()
        self.data = datasets.ImageFolder(root=root + '/' + split, transform=None)
        self._split = split
        self._num_examples = num_examples
        self._transform = transform
        self._processor = processor
        if self._split in ['imagenet-r', 'renditions']:
            self.valid_indices = r_indices
        elif self._split == 'imagenet-a':
            self.valid_indices = a_indices
        if self._num_examples is not None:
            if self._num_examples > len(self.data):
                raise ValueError('num_examples can be at most the dataset size {len(self.data)}')
            rng = np.random.RandomState(seed=seed)
            self._data_indices = rng.permutation(len(self.data))[:num_examples]

    def __getitem__(self, i):
        if self._num_examples is not None:
            i = self._data_indices[i]
        x, y = self.data[i]
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        if self._processor is not None:
            x = self._processor(images=x, return_tensors="pt")["pixel_values"].squeeze(0)
        if self._split == 'renditions' or self._split == 'imagenet-r':
            y = r_indices[y]
        elif self._split == 'imagenet-a':
            y = a_indices[y]
        elif self._split == 'v2' or self._split == 'imagenetv2-matched-frequency-format-val':
            y = v2_indices[y]
        return x, y

    def __len__(self) -> int:
        return len(self.data) if self._num_examples is None else self._num_examples

def bernoulli_kl(logit1, logit2, eps=1e-7, reduction="batchmean"):
    p = torch.sigmoid(logit1)
    q = torch.sigmoid(logit2)

    p = p.clamp(min=eps, max=1 - eps)
    q = q.clamp(min=eps, max=1 - eps)

    # KL divergence between Bernoulli(p) and Bernoulli(q)
    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    if reduction == "batchmean":
        return kl.mean()

def get_failure_list(failure_path, dataset, model):
    if os.path.exists(failure_path):
        with open(failure_path, 'rb') as f:
            failure_indices = pickle.load(f)
    else:
        failure_indices = []
        model.eval()
        with torch.no_grad():
            for idx, (image, label, _) in enumerate(dataset):
                image = image.unsqueeze(0).cuda()  # add batch dim
                label = torch.tensor(label).cuda()

                output = model(image).logits
                pred = output.argmax(dim=1)

                if pred.item() != label.item():
                    failure_indices.append(idx)

        # Save to pickle
        with open(failure_path, 'wb') as f:
            pickle.dump(failure_indices, f)

    # Create filtered failure-case dataset
    return failure_indices