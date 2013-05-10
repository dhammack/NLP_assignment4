#import count_freqs
import sys
import eval_gene_tagger as ev
import itertools
from collections import defaultdict
import Queue
import string
import re
import multiprocessing
import time
import pylab as pl
import numpy as np

def getsentences(path):
    sents = [[]]
    with open(path,"r") as f:
        for line in f:
            items = line.split(' ')
            if items[0].replace('\n','') == "":
                #end of sent, add STOP and *,*
                #sents[-1].append("-STOP-")
                #sents[-1].insert(0,"*")
                #sents[-1].insert(0,"*")
                sents.append([])
            else:
                sents[-1].append(items[0].replace('\n', ''))
    return [s for s in sents if len(s) > 0]

def states(index):
    if index == -2 or index == -1:
        return ['*']
    else:
        return ['O','I-GENE']
    
def evalTags(keyFile,predictionsFile):
    gs_iterator = ev.corpus_iterator(file(keyFile))
    pred_iterator = ev.corpus_iterator(file(predictionsFile), with_logprob = False)
    evaluator = ev.Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    evaluator.print_scores()


def tagFile(infile, outfile):
    #read the input
    sents = getsentences(infile)
    with open(outfile,"w") as out:
        for sentence in sents:
            #tag and then output.
            tags = viterbi_GLM(sentence, params)
            for tag,word in zip(tags,sentence):
                out.write(word + " " + tag + "\n")
            out.write('\n')


def features(history, tag): #history is a 4-tuple: (tag-2, tag-1, words, index)
    #generates a map from strings to counts
    g = {}

    #the following features are defined for all tags S,U,V, and words r
    #and now j in {1,2,3}

    #gTRIGRAM:S:U:V -> 1 if t_-2=S and t_-1=U and t=V
    g[ (history[0],history[1],tag) ] = 1
    g[ (0, history[1], tag) ] = 1
    g[ (1, history[0],tag)] = 1
    
    if not 0 <= history[3] < len(history[2]):
        return g
    
    #try:
    word = history[2][history[3]]
    #except:
    #    return g

    #gTAG:U:r -> 1 if word = r and tag = U
    g[ (word,tag) ] = 1

    g[ (str.lower(word),tag) ] = 1
    
    #gNUM:U:r -> 1 if word contains numeric and tag=N
    b = containsnum(word)
    if b: g[ (tag, 'N') ] = 1
    
    #gSUFF:U:j:v -> 1 if u = suffix(word,j) and t=V
    for j in xrange(1,4):
        g[ (word[-j:], j, tag) ] = 1

    #gINITIALCAPS:word:V -> 1 if word starts with capital letter, tag=V
    b = str.isupper(word[0])
    if b: g[ (tag, 'U') ] = 1
    
    #gWORDLEN:word:V -> 1 if word length = L and tag = V
    L = len(word)
    g[ ('len',L,tag) ] = 1

    #gPREV:word:V -> 1 if prevword == word, tag=V
    if 0 <= history[3]-1 < len(history[2]):
        g[ (history[2][history[3]-1], tag) ] = 1
        g[ (history[2][history[3]-1], history[1], tag, 'prevpair') ] = 1
   

    #prefix features
    for j in xrange(1,4):
        g[ (word[:j], j, tag) ] = 1
        
    #count nums
    #k = sum(1 for i in word if str.isupper(i))
    #n = sum(1 for i in word if containsnum(i))
    #g[ (n, tag,'numct') ] = 1
    #g[ (k, tag, 'upperct') ] = 1
    return g

_digits = re.compile('\d')
def containsnum(word):
    return bool(_digits.search(word))

def indicator_update(old, to_add):
    for i in to_add.iterkeys():
        if i not in old:
            old[i] = 1
            
def dictupdate(old,to_add):
    for i in to_add.iterkeys():
        if i in old:
            old[i] += to_add[i]
        else:
            old[i] = to_add[i]

def global_features(sentence, tags):
    #t_tags = tags[::]
    #t_tags.insert(0,'*')
    #t_tags.insert(0,'*')
    result = {}
    
    for index in xrange(0,len(tags)-2):
        dictupdate(result,
                   (features((tags[index],tags[index+1],sentence,index+2),
                               tags[index+2])))

    return result

    
def viterbi_GLM(sent, gparams):
    #v = params.copy()
    
    pi = { (-1,'*','*'):0 } #notes say this should be zero. check.
    bp = {}# (-1,'*','*'):'O' } #not sure if should be here.  
    #for index,word in enumerate(sent):
    for index in xrange(0,len(sent)):
        for u,s in itertools.product(states(index-1), states(index)):

            max_state, max_prob = 'O',-1000000.0
            for t in states(index-2):
                f = features( (t,u,sent,index), s)
                cur_prob = pi[(index-1,t,u)] + dot(f, gparams )
                
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_state = t

            pi[(index,u,s)] = max_prob
            bp[(index,u,s)] = max_state

    #for i in sorted(pi.iteritems()):
    #    print 'pi[',i[0],']:',i[1]

    try:
        n = len(sent)-1
        tags = ['O'] * (n+1)
        max_prob, max_s, max_u = -10000000.0, 'O', 'O'
        for u,s in itertools.product(['O','I-GENE'],['O','I-GENE']):
            cur_prob = pi[(n,u,s)] + dot(features( (u,s,sent,n+1), 'STOP'), gparams)

            if cur_prob > max_prob:
                max_s = s
                max_u = u
                max_prob = cur_prob
            
        tags[n-1] = max_u
        tags[n] = max_s
    except:
        print 'error tagging sentence:'
        print sent
        print n
        raw_input()
    for index in range(n+1)[-3::-1]:
        tags[index] = bp[ (index+2,tags[index+1],tags[index+2]) ]

    return tags
                                   

def load_params(filepath):
    global params
    params.clear()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.replace('\n','')
            cols = line.split(' ')
            items = cols[0].split(':')

            if items[0] == 'TAG':
                params[ (items[1],items[2]) ] = float(cols[1])
            elif items[0] == 'TRIGRAM':
                params[ (items[1],items[2],items[3]) ] = float(cols[1])


def dot(feature_vect, pglobal):
    return sum(pglobal[key] * val for key, val in feature_vect.iteritems())


def update_step(v,best,gold):
    #v = v + gold - best
    for key in gold.iterkeys():
        if key in v:
            v[key] += gold[key]
        else:
            v[key] = gold[key]
                
    for key in best.iterkeys():
        if key in v:
            v[key] -= best[key]
        else:
            v[key] = -1 * best[key]
    return v
    
def train_model(trainpath, N, M): #M= number of shards/threads
    global params
    tempsents = []
    with open(trainpath,"r") as f:
        cur_sent = []
        cur_tags = []
        for line in f:
            if line == '\n':
                tempsents.append( (cur_sent, cur_tags) ) #list of tuples of lists
                cur_sent = []
                cur_tags = []
                continue
            
            items = line.split(' ')
            items[1] = items[1].replace('\n','')
            cur_sent.append(items[0])
            cur_tags.append(items[1])

    sents = [s for s in tempsents if len(s[0]) > 1]
    trainsize = len(sents)
    shardsize = trainsize / M
    shard = []
    for k in xrange(M):
        shard.append(sents[k*shardsize:(k+1)*shardsize])
    L = multiprocessing.Lock()
    for T in xrange(N):
        #each training iteration.
        print 'iteration:',T+1
        if M > 1:
            out_q = []
            tp = multiprocessing.Pool()
            with L: pcopy = params.copy()
            for sh in shard:
                tp.apply_async(train_map, args=(sh,pcopy),
                                 callback= lambda x: out_q.append(x))
            tp.close()
            tp.join()
            v = {}
            for vector in out_q:
                dictupdate(v,vector)
            dictupdate(params, v)
            #vect_multiply(params, 1.0/float(M))
            #v = params.copy()
            #params.clear()
            #params = {key: float(val)/float(M) for key,val in v.iteritems()}
            #tagFile("gene.dev","gene_dev_t7.out")
            #eval_model('gene.key', 'gene_dev_t7.out',T+1)
        else:
            print 'singlethread:'
            #params = train_map(sents,params)
            
            with L: pcopy = params.copy()
            train_map(sents,pcopy)
            with L: params = pcopy.copy()
            tagFile("gene.dev","gene_dev_t7.out")
            eval_model('gene.key', 'gene_dev_t7.out',T+1)
            

def vect_multiply(vector, scalar):
    for key,val in vector.iteritems():
        vector[key] = val * scalar
        
def train_map(sents, weights):
    #L = multiprocessing.Lock()
    #with L:
    #v = weights.copy()
    for x,y in sents:
        #1 - compute best tagging.
        z = viterbi_GLM(sent=x, gparams=weights)
        #2 - compute best tagging feature vector.
        feat_best = global_features(sentence=x,tags=z)
        #3 - compute gold tagging feature vector.
        feat_gold = global_features(sentence=x,tags=y)
        #4 - update weights. v = v + f(x,y) - f(x,z)
        weights = update_step(v=weights, gold=feat_gold, best=feat_best)

    #with L:
    #copyto = weights.copy()
    return weights

def eval_model(keyFile, predictionsFile, iterct):
    #record accuracy and settings.
    gs_iterator = ev.corpus_iterator(file(keyFile))
    pred_iterator = ev.corpus_iterator(file(predictionsFile), with_logprob = False)
    evaluator = ev.Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    (prec, rec, f) = evaluator.get_scores()
    datapoints.append((iterct, prec, rec, f))

def plot_results(datapoints): #datapoints: (iteration, prec, rec, f1), 
    i = [d[0] for d in datapoints]
    p = [d[1] for d in datapoints]
    r = [d[2] for d in datapoints]
    f = [d[3] for d in datapoints]
    pl.plot(i,p,color='green')
    pl.plot(i,r,color='blue')
    pl.plot(i,f,color='red')
    pl.show()
    
if __name__ == '__main__':
    params = defaultdict(float) #global parameter vector, v.
    datapoints = []
    #load_params('tag.model')
    t1 = time.time()
    train_model(trainpath='gene.train',N=5,M=1)
    print 'time:',time.time()-t1
    tagFile("gene.dev","gene_dev_t7.out")
    #plot_results(datapoints)
    evalTags("gene.key","gene_dev_t7.out")
    #tagFile('gene.test','gene_test_p4.out')
    #print viterbi_GLM(['There','was','gene','.'])

    #trainpreprocess("gene.train")
    #getCounts("gene.train.mod","counts_mod.txt")
    #(pairLookup,wordLookup,unigramLookup,bigramLookup,trigramLookup) = initialize("counts_mod.txt")

    #tagFile("gene.debug","gene_debug.p2.out")


