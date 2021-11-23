def stats(path):
  print(path)
  with open(path,"r") as f:
    lines = f.readlines()
    print("Total lines:",len(lines))

  words = 0
  len_of_sentences = []
  for i,line in enumerate(lines):
    words += 1
    if line == '\n':
      len_of_sentences.append(words)
      words = 0  

  print(sorted(len_of_sentences,reverse=True)[0:20],"\n")


def remove_long(path):
  print(path)
  new_path = path.split('/')[0]+'/new_'+path.split('/')[1]
  print(new_path)

  with open(path,"r") as f:
    lines = f.readlines()
    f.close()

  words = []
  for i,word in enumerate(lines):
    words.append(word)
    # print(word)
    if word == '\n':
      if len(words) <= 510: 
        with open(new_path,"a+") as f:
          for w in words:
            f.write(w)
          f.close()
      else:
        print(len(words),i)
      words = []
  print()

def main():

  remove_long("NER_Dataset/train.txt")  
  remove_long("NER_Dataset/dev.txt")  
  remove_long("NER_Dataset/test.txt")  

  stats("NER_Dataset/train.txt")  
  stats("NER_Dataset/dev.txt")  
  stats("NER_Dataset/test.txt")  

  stats("NER_Dataset/new_train.txt")  
  stats("NER_Dataset/new_dev.txt")  
  stats("NER_Dataset/new_test.txt")  

if __name__ == '__main__':
    main()