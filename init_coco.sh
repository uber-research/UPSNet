# install coco panoptic api

if [ ! -d lib/dataset_devkit/panopticapi]; then
    git clone https://github.com/cocodataset/panopticapi lib/dataset_devkit/panopticapi
fi

python init_coco.py

PYTHONPATH=$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_train2017_stff.json --segmentations_folder data/coco/annotations/panoptic_train2017 --semantic_seg_folder data/coco/annotations/panoptic_train2017_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_coco_categories_stff.json
PYTHONPATH=$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_val2017_stff.json --segmentations_folder data/coco/annotations/panoptic_val2017 --semantic_seg_folder data/coco/annotations/panoptic_val2017_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_coco_categories_stff.json
