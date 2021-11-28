import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_dataset_xml(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height

            dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)


def load_dataset_yolotxt(path, txt, labels):
    """
    :param path: directory include text file, and label folder
    :param txt: a text file claim train image paths
    :param labels: folder name which saves rectangular params within each txt file
    :return: dataset as numpy array format, (N, 2)
    """

    train_images_path = os.path.join(path, txt)  # a txt file contains all train images name and path
    train_labels_path = os.path.join(path, labels)
    label_file = []

    with open(train_images_path, 'r') as t:
        img_path = t.readlines()
        for line in img_path:
            tmp_path = line.rsplit('/', 1)[-1]
            label_file.append(tmp_path[:-5] + '.txt')

    print('label count: {}'.format(len(label_file)))
    dataset = []

    for label in label_file:
        with open(os.path.join(train_labels_path, label), 'r') as f:
            txt_content = f.readlines()

        for line in txt_content:
            line_split = line.split(' ')
            roi_with = float(line_split[len(line_split)-2])
            roi_height = float(line_split[len(line_split)-1])
            if roi_with == 0 or roi_height == 0:
                continue
            dataset.append([roi_with, roi_height])
            # print([roi_with, roi_height])

    return np.array(dataset)


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]  # row = 2086

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    # make result repeatable
    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # random select any box as initial clusters

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)  # calculate iou, and 1-iou define distance

        nearest_clusters = np.argmin(distances, axis=1)  # min distance == max iou

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def plot_anchors(x, y):
    for i in range(len(x)):
        plt.scatter(x[i], y[i], cmap=plt.cm.Paired, s=20)
        # plt.plot(x[i], y[i])
    plt.show()


# cluster calculated anchors to replace default anchor
def cluster_anchors(X, n_cluster, random_state=0):
    out = KMeans(n_cluster, random_state = random_state).fit(X)
    centers = out.cluster_centers_
    return centers










