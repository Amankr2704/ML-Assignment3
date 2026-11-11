from pathlib import Path
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'generated_figures'
OUT.mkdir(exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved', name)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
for name in ['shallow','deep']:
    h = pd.read_csv(ROOT/'results'/f'{name}_history.csv')
    axes[0].plot(h.epoch,h.train_acc,label=f'{name} train')
    axes[0].plot(h.epoch,h.val_acc,'--',label=f'{name} validation')
    axes[1].plot(h.epoch,h.train_loss,label=f'{name} train')
    axes[1].plot(h.epoch,h.val_loss,'--',label=f'{name} validation')
for ax,title in zip(axes,['Accuracy','Cross-entropy loss']):
    ax.set(xlabel='Epoch',ylabel=title); ax.grid(alpha=.25); ax.legend()
save(fig,'q1a_learning_curves.png')

h=pd.read_csv(ROOT/'results/memorization_history.csv')
fig,ax=plt.subplots(figsize=(9,5))
for col,label in [('clean_train_acc','Clean training'),('corrupted_true_acc','Corrupted: true labels'),('corrupted_wrong_acc','Corrupted: wrong labels'),('val_acc','Clean validation')]: ax.plot(h.epoch,h[col],label=label)
ax.axhline(.1,color='gray',linestyle=':',label='10-class chance reference')
ax.set(xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.02)); ax.grid(alpha=.25); ax.legend()
save(fig,'q1b_memorization.png')

fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
for ax,condition in zip(axes,['clean','shuffled','grayscale']):
    for model in ['MLP','CNN']:
        h=pd.read_csv(ROOT/'results_q2'/f'{model}_{condition}_history.csv')
        ax.plot(h.epoch,h.train_acc,label=f'{model} train')
        ax.plot(h.epoch,h.val_acc,'--',label=f'{model} validation')
    ax.set(title=condition,xlabel='Epoch'); ax.grid(alpha=.25)
axes[0].set_ylabel('Accuracy'); axes[-1].legend()
save(fig,'q2_learning_curves.png')

fig,axes=plt.subplots(1,2,figsize=(11,4))
for size in [32,128,512]:
    h=pd.read_csv(ROOT/'results_q3'/f'clean_{size}_history.csv')
    axes[0].plot(h.epoch,h.train_mse,label=str(size))
    axes[1].plot(h.epoch,h.val_mse,label=str(size))
for ax,title in zip(axes,['Training MSE','Validation MSE']):
    ax.set(title=title,xlabel='Epoch',ylabel='MSE per pixel'); ax.grid(alpha=.25); ax.legend(title='Bottleneck')
save(fig,'q3_learning_curves.png')

sweep=pd.read_csv(ROOT/'results_q4/svm_sweep.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for kernel in ['linear','rbf','poly']:
    h=sweep[sweep.Kernel==kernel]
    axes[0].semilogx(h.C,h['Validation accuracy'],marker='o',label=kernel)
    axes[1].semilogx(h.C,h['Support vector fraction'],marker='o',label=kernel)
for ax,title in zip(axes,['Validation accuracy','Support vector fraction']):
    ax.set(title=title,xlabel='C'); ax.grid(alpha=.25); ax.legend()
save(fig,'q4_svm_sweep.png')

# Images in the saved Q4 example table refer to the original CIFAR-10 batch indices.
data_dir=ROOT/'data/cifar-10-batches-py'
if not (data_dir/'test_batch').exists():
    print('CIFAR-10 not downloaded here; skipping Q4 image grids. Run Q4.ipynb to generate them.')
else:
    def read_batch(name):
        with (data_dir/name).open('rb') as f: batch=pickle.load(f,encoding='bytes')
        return batch[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    train_images=np.concatenate([read_batch(f'data_batch_{i}') for i in range(1,6)])
    test_images=read_batch('test_batch')
    examples=pd.read_csv(ROOT/'results_q4/pair_support_and_boundary_examples.csv')
    for pair in examples.Pair.unique():
        rows=examples[examples.Pair==pair]
        fig,axes=plt.subplots(2,6,figsize=(15,5))
        for row_num,(kind,images) in enumerate([('train support vector',train_images),('near-boundary test',test_images)]):
            for ax,(_,item) in itertools.zip_longest(axes[row_num],rows[rows.Type==kind].iterrows(),fillvalue=(None,None)):
                if ax is None: continue
                ax.axis('off')
                if item is None: continue
                ax.imshow(images[int(item['Dataset index'])])
                label=str(item['True class'])
                if row_num: label+=' to '+str(item['Prediction'])
                ax.set_title(f"{label}\nscore={item['Pairwise score']:.3f}",fontsize=9)
        fig.suptitle(pair)
        save(fig,'q4_boundary_'+pair.replace(' vs ','_').replace(' ','_')+'.png')
