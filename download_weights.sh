if [ ! -f model/upsnet_resnet_101_dcn_coco_270000.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/upsnet_resnet_101_dcn_coco_270000.pth -o model/upsnet_resnet_101_dcn_coco_270000.pth
fi

if [ ! -f model/upsnet_resnet_50_coco_90000.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/upsnet_resnet_50_coco_90000.pth -o model/upsnet_resnet_50_coco_90000.pth
fi

if [ ! -f model/upsnet_resnet_101_cityscapes_w_coco_3000.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/upsnet_resnet_101_cityscapes_w_coco_3000.pth -o model/upsnet_resnet_101_cityscapes_w_coco_3000.pth                                                                                                                                    
fi

if [ ! -f model/upsnet_resnet_101_coco_pretrained_for_cityscapes.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/upsnet_resnet_101_coco_pretrained_for_cityscapes.pth -o model/upsnet_resnet_101_coco_pretrained_for_cityscapes.pth
fi

if [ ! -f model/upsnet_resnet_50_cityscapes_12000.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/upsnet_resnet_50_cityscapes_12000.pth -o model/upsnet_resnet_50_cityscapes_12000.pth
fi

