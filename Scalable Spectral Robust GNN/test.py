from configs.data_config import data_args
from data_process import dataRead
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = dataRead(data_args.data_root, data_args.data_name)
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(data.x)
    y = data.y.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig('test.png')
    plt.show()