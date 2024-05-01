import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

basic_img = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
)
aug_img = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
)
basic_open = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ]
)
aug_open = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ]
)
basic_M = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToTensorV2(),
    ]
)

aug_M = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.8, hue=0.2, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=2, p=0.5),
        A.GaussianBlur(p=0.4),
        A.ImageCompression(p=0.1),
        A.ToGray(p=0.3),
        A.Sharpen((0.2, 0.5), (0.5, 1.0), p=0.2),
        A.ChannelShuffle(p=0.4),
        A.Downscale(0.5, 0.5, p=0.2),
        A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToTensorV2(),
    ]
)
