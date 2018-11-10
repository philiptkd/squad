import matplotlib.pyplot as plt

def get_word_lengths(filename):
    word_lens = []    
    with open(filename, 'r') as f:
        for line in f:
            word_list = line.split()
            for word in word_list:
                word_lens.append(len(word))
    return word_lens
   
plt.subplot(221)
context_lens = get_word_lengths('train.context')
plt.hist(context_lens, range=(1,20))
plt.title('Context Word Lengths')

plt.subplot(222)
question_lens = get_word_lengths('train.question')
plt.hist(question_lens, range=(1,20))
plt.title('Question Word Lengths')

plt.subplot(223)
context_lens = get_word_lengths('train.context')
plt.hist(context_lens, range=(20,50))
#plt.title('Context Word Lengths')

plt.subplot(224)
question_lens = get_word_lengths('train.question')
plt.hist(question_lens, range=(20,50))
#plt.title('Question Word Lengths')

plt.show()

