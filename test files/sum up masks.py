from ultralytics import YOLO
import cv2
import numpy as np

model  = YOLO("../yolo-weights/yolov8x-seg.pt")
result1 = model.predict('../example screenshots/pic3.png', retina_masks=True)

masks = result1[0].masks.cpu()
all_masks = masks.data.numpy()
# cv2.imshow('winname2',(all_masks[1]*255).astype("uint8"))
summed_masks = np.sum(all_masks,axis=0)
normalized_mask = 255 - (summed_masks > 0).astype(np.uint8) * 255
cv2.imwrite('masks.jpg', normalized_mask)





# shaped_arr_mask = arr.reshape(-1, arr.shape[-1])



# path=model.export(format='openvino')
# model_2 = YOLO(path)
# result2 = model_2.predict('../example screenshots/pic3.png', retina_masks=True)
# result2[0].show()