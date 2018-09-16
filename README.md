# Code and data for Aspect Based Sentiment Analysis with Gated Convolutional Networks

```
@inproceedings{DBLP:conf/acl/LiX18,
  author    = {Wei Xue and Tao Li},
  title     = {Aspect Based Sentiment Analysis with Gated Convolutional Networks},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2018, Melbourne, Australia, July 15-20, 2018, Volume
               1: Long Papers},
  pages     = {2514--2523},
  year      = {2018},
  crossref  = {DBLP:conf/acl/2018-1},
  url       = {https://aclanthology.info/papers/P18-1234/p18-1234},
  timestamp = {Thu, 12 Jul 2018 14:15:56 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/acl/LiX18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


# Instructions:
Download glove or word2vec file and change the path in w2v.py correspondingly.

## ACSA
python -m run -lr 1e-2 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect    -embed_file glove  -r_l r  -epochs 13

python -m run -lr 1e-2 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect    -embed_file glove  -r_l r  -year 14 -epochs 5

## ATSA
python -m run -lr 5e-3 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect  -embed_file glove  -r_l r -year 14 -epochs 6 -atsa

python -m run -lr 5e-3 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect  -embed_file glove  -r_l l -year 14 -epochs 5 -atsa


