import cv2
import numpy as np


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    """
        test everything
    """
    import os
    import mrcnn.model as modellib
    from mrcnn.config import Config


    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
    MODEL_WEIGHT_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_animal_0018.h5')

    if not os.path.exists(MODEL_WEIGHT_PATH):
        print('Did not find this weight' + MODEL_WEIGHT_PATH)
        exit()

    class BottleConfig(Config):
        NAME = "animal"

        IMAGES_PER_GPU = 2

        NUM_CLASSES = 1 + 2


    class InferenceConfig(BottleConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_WEIGHT_PATH, by_name=True)
    class_names = ['BG', 'dog', 'cat']

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        for item in r['class_ids']:
            print(class_names[item])

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()