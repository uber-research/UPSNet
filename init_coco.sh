python init_coco.py

sed -i 's/semantic = np.zeros(pan.shape, dtype=np.uint8)/semantic = np.ones(pan.shape, dtype=np.uint8) * 255/g' lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py

PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_train2017_stff.json --segmentations_folder data/coco/annotations/panoptic_train2017 --semantic_seg_folder data/coco/annotations/panoptic_train2017_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_coco_categories_stff.json
PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file data/coco/annotations/panoptic_val2017_stff.json --segmentations_folder data/coco/annotations/panoptic_val2017 --semantic_seg_folder data/coco/annotations/panoptic_val2017_semantic_trainid_stff --categories_json_file data/coco/annotations/panoptic_coco_categories_stff.json
