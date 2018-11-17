# Network architecture
The network is a pre-trained Inception v3 net, with the last layers replaced.


# Multiprocessing
One of the bottlenecks during training was the loading and augmentation of data. Using multiprocessing with 12 hyperthreaded cores helped us to speed up the learning process significantly.

See:
https://github.com/stratospark/keras-multiprocess-image-data-generator