import os
import json
import cv2 as cv
import PIL.Image
import numpy as np
from labelme import utils
import csv
import io
import PIL.ImageDraw
import os.path as osp
import cv2


def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values
def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap
# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled
    # change to Binary image

    return lbl_viz
def draw_label(label, img=None, label_names=None, colormap=None):
    import matplotlib.pyplot as plt
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    if colormap is None:
        colormap = label_colormap(len(label_names))

    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        if label_name.startswith('_'):
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        #code for include title for annotating image
        #start code
    #     plt_titles.append('{name}'
    #                       .format(value=label_value, name=label_name))
    # plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)
    #end code
    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)

    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out



json_file=r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_train'   #json path
save_path=r'C:\Users\nhatn\PycharmProjects\KTLTPython\image_train_segment'
list_path = os.listdir(json_file)


for i in range(0, len(list_path)):
    path = os.path.join(json_file, list_path[i])
    if os.path.isfile(path) and ".json" in path:
        data = json.load(open(path,encoding="utf8", errors='ignore'))
        img = utils.img_b64_to_arr(data['imageData'])
        lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
        lbl_dict = {'_background_': [0, 0, 0], 'Name': [128, 0, 0], 'Year School':[0,0,128],'Major':[0,128,0],'ID':[128,128,0],'Image':[128,0,128]}
        captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
        # print(captions)
        keys = [i for i in lbl_names]
        img = np.dstack((lbl, lbl, lbl))
        for i in range(len(lbl)):
            for j in range(len(lbl[0])):
                img[i][j] = lbl_dict[keys[lbl[i][j]]]
        img = np.array(img, dtype=np.float32)

        out_dir = osp.basename(path).split('.json')[0]
        save_file_name = out_dir+"_converted.png"
        os.chdir(save_path)
        cv2.imwrite(save_file_name,img)
