# -*- coding: utf-8 -*-
"""
==================================
Color Quantization using K-Means
==================================
"""


print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
from appJar import gui


def press(button):
    if button == "Cancel":
        app.stop() ##bbuton cancel aplikasi berhenti
    else:
        image = app.getEntry("image")
        print("Image path:", image)## untuk mengambil gambar
        n_colors = int(app.getEntry("Color"))##untuk memasukan jumlah warna ke aplikasi
        if n_colors == 0: ##dan jika nilai warna 0 maka eror
            app.errorBox("Error", "Please, type number of colors!")
        # return image

        # photos = load_sample_image(image)
        photos = Image.open(image)
        ff = Image.open(image)


        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        photos = np.array(photos, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(photos.shape)
        assert d == 3
        image_array = np.reshape(photos, (w * h, d))



        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))

        print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        print("done in %0.3fs." % (time() - t0))


        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image"""
            d = codebook.shape[1]
            image = np.zeros((w, h, d))
            label_idx = 0
            for i in range(w):
                for j in range(h):
                    image[i][j] = codebook[labels[label_idx]]
                    label_idx += 1
            return image

        # Display all results, alongside original image

        plt.figure("Original image")
        plt.clf()
        ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        outputor = BytesIO()
        ff.save(outputor, 'PNG')  # a format needs to be provided
        contentsor = outputor.getvalue()
        outputor.close()
        image_filesizeor = len(contentsor)
        plt.title(image_filesizeor)
        plt.imshow(photos)

        plt.figure("Quantized image (K-Means)")
        plt.clf()
        ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        kmean = recreate_image(kmeans.cluster_centers_, labels, w, h)
        imgk = Image.fromarray(kmean, 'RGB')
        output = BytesIO()
        imgk.save(output, 'PNG')  # a format needs to be provided
        contents = output.getvalue()
        output.close()
        image_filesize = len(contents)
        plt.title(image_filesize)
        plt.imshow(kmean)
        plt.show()






app = gui("Color Quantization using K-Means", "700x200")
app.setFont(20)
app.addLabel("title", "Welcome", 0, 0)
app.setFont(14)
app.addLabel("choose", "Select Image", 1, 0)
app.addFileEntry("image", 1, 1)
app.setFont(14)
app.addLabel("color_head", "Number of Colors", 2, 0)
app.addNumericEntry("Color", 2, 1)
app.setBg("black")

app.setLabelBg("title", "red")
app.setLabelBg("choose", "green")
app.setLabelBg("color_head", "blue")

# app.addLabelEntry("image")
# print("salam")

app.setFocus("image")
app.addButtons(["Submit", "Cancel"], press, 3, 1)

app.addLabel("footer", "TUGAS POLA", 3, 0)
app.setLabelBg("footer", "white")

app.go()

