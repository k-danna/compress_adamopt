#! /usr/bin/env python3

import time, array
import numpy as np

def timing(f):
    def wrap(*args):
        start = time.time()
        r = f(*args)
        elapsed = time.time() - start
        print('[*] %s took %0.3f s' % (f.__name__, elapsed))
        return r
    return wrap

def next_block(f, bsize, dtype=np.uint8):
    block = np.asarray([x for x in f.read(bsize)], dtype=dtype)
    need = bsize - len(block)
    if need and len(block) > 0:
        block = np.append(block, np.zeros(need, dtype=dtype), axis=0)
    return block

@timing
def compress(f_name, bsize=1024*1):
    out_name = 'out.cmp'
    #with open(f_name, 'rb') as f, open(out_name, 'ab') as o:
    with open(f_name, 'rb') as f:
        #init weights
        block = next_block(f, bsize)
        n = bsize
        w = {
            'w1': np.random.normal(0.0, 0.01, (bsize, n)),
            'w2': np.random.normal(0.0, 0.01, (n, bsize)),
            'b1': np.random.normal(0.0, 0.01, (n,)),
            'b2': np.random.normal(0.0, 0.01, (bsize,)),
        }
        w = fit(block, w)
    
        while True:
            #read data block
            block = next_block(f, bsize)
            if not len(block):
                break

            #compress the block
            w = fit(block, w)

            #FIXME: debug
            #break

            #write compressed block
            #s = array.array('B', block).tostring()
            #with open('out.temp', 'ab') as t:
            #    t.write(block)
                
    #return name of written file
    return ''

@timing
def fit(x, w): #given uint8 array
    x = np.expand_dims(x, axis=0).astype(np.float32, copy=False)

    #forward
    loss = 1.0
    lr = 1e-3
    clip = 5
    t = 0
    out = []
    y = np.asarray([])
    m = {k: 0.0 for k in w}
    v = {k: 0.0 for k in w}
    while not np.array_equal(x, y.astype(np.uint8)):
        #forward
        h = np.matmul(x, w['w1']) + w['b1']
        relu = np.clip(h, 0.0, None)
        y = np.matmul(h, w['w2']) + w['b2']
        o = np.floor(y) - x
        out.append(np.squeeze(o))

        loss = 0.5 * np.sum(np.square(o))
        #print('loss:', loss)

        #backward
        g = {}
        g['y'] = o
        g['w2'] = np.matmul(relu.T, g['y'])
        g['b2'] = np.squeeze(g['y'])
        g['relu'] = np.matmul(g['y'], w['w2'].T)
        g['h'] = g['relu']
        g['h'][h < 0] = 0
        g['w1'] = np.matmul(x.T, g['h'])
        g['b1'] = np.squeeze(g['h'])

        #update
        for k in w:
            assert(w[k].shape == g[k].shape)

        #update
        t += 1
        #FIXME: names are all wonky
        b1, b2, e = 0.9, 0.999, 1e-8
        lr_t = (lr / np.sqrt(t)) * np.sqrt(1 - np.power(b2, t)) / (1 - np.power(b1, t))
        #lr_t = lr * np.sqrt(1 - np.power(b2, t)) / (1 - np.power(b1, t))
    
        #lr *= decay
        for k in w:
            #g[k] = np.clip(g[k], -clip, clip)
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * np.square(g[k])
            w[k] -= lr_t * m[k] / (np.sqrt(v[k]) + e)
            #w[k] -= lr * np.clip(g[k], -clip, clip)
        
    print('steps:', t)

    x = np.squeeze(x)
    y = np.squeeze(y.astype(np.uint8))

    assert(np.array_equal(x, y))

    out = np.asarray(out)
    out_small = out.astype(np.int16)
    assert(np.array_equal(out, out_small))

    #with open('out.tmp', 'ab') as f:
    #    np.save(f, out)


    return w

#@timing
#def decompress(f):
#    data = DataWrapper(file_name)
#    while True:
#        block = data.read_block()
#        if not block:
#            break
#        #print('\n', block)

@timing
def main():

    np.random.seed(42)

    #currently available datasets
    data_files = [
        'data/shakespeare.txt',  #1.1 mb
        'data/grimms.txt',       #541 kb
        'data/bible.txt',        #4.4 mb
        'data/mobydick.txt',     #1.2 mb
        'data/tweets.txt',       #3.8 mb
        'data/tweets_clean.txt', #3.6 mb
    ]

    #get data handle
    f_name = data_files[1]

    #print('[*] compressing %s' % file_name)
    f_name = compress(f_name)
    #print('[*] decompressing %s' % file_name)
    #decompress(f_name)


if __name__ == '__main__':
    main()

