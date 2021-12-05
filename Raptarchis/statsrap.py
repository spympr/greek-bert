import pandas as pd
from torch import sub

train_df = pd.read_csv('../Raptarchis/train.csv')
dev_df   = pd.read_csv('../Raptarchis/dev.csv')
test_df  = pd.read_csv('../Raptarchis/test.csv')

volume_labels = set()
for elem in train_df.volume.unique():
  volume_labels.add(elem)

for elem in dev_df.volume.unique():
  volume_labels.add(elem)

for elem in test_df.volume.unique():
  volume_labels.add(elem)

chapter_labels = set()
for elem in train_df.chapter.unique():
  chapter_labels.add(elem)

for elem in dev_df.chapter.unique():
  chapter_labels.add(elem)

for elem in test_df.chapter.unique():
  chapter_labels.add(elem)

subject_labels = set()
for elem in train_df.subject.unique():
  subject_labels.add(elem)

for elem in dev_df.subject.unique():
  subject_labels.add(elem)

for elem in test_df.subject.unique():
  subject_labels.add(elem)

print("Volume:",len(volume_labels))
print("Chapter:",len(chapter_labels))
print("Subject:",len(subject_labels))