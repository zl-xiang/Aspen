# ASPen: ASP-based Collective Entity Resolution

## Environment Installation
Although it is suffice to install directly with `pip`, we recommand using a `conda` environment as a container.
To do so, one could execute optionally the following:
- create a `conda` environment:
```
conda create -n aspen python=3.9.7
```
- activate the environment:
```
conda activate aspen
```

The running environment is maintained using `pip` based on `Python 3.9.7`. To install, directing first to the source folder `aspen/src` then executing the following:

```
pip install -r requirements.txt
```



## Experiment Running Instruction

A general explanantion of the command is the following:


```
python mains_explain_.py \
    --asprin "enable asprin for maximal solution"
    --cached "caching results or not"
    --enumerate 1 "enumerate the number of maximal solutions required if maxsol"
    --data "choosing version to use if music dataset, available choices:  "10", "30", "50", "50-corr"
    --schema "choosing dataset schema from {"dblp", "cora", "imdb", "music", "pokemon"}"
    --main "the main |ER control function, enable if requiring ub/lb or maxsol"
    --rec-track "procedure to compute levels"
    --getsim "procedure to compute similarity facts on a dataset"
    --presimed "to load precomputed similarity facts"
    -m "directory where the optimisation statement locates" \
    -l "a list of logic program as input specification" \
    --typed_eval "activate evaluation for multi-relational dataset"
    --ternary "enable ternary representation of a specification"
    --ub "derive ub solution"
    --lb "derive lb solution"
    --no_show "projection of merges without relation type (on dblp and cora)"
    --debug "run in debug mode"
    --pos-merge "compute all PM input: all, verify a merge input c,c'"
    --trace-merge "input the merge to provide proof tree"
    --attr "input what relation and attribute the merge is belong to when explaining, e.g. release id"
```




### 1 Similarity Filtering

```
# dblp
python -u mains_explain_.py -c -l ./experiment/dblp/dblp.lp  --getsim --typed_eval --no_show --ternary --schema dblp
# cora
python -u mains_explain_.py -c -l ./experiment/cora/cora.lp  --getsim --typed_eval --no_show --ternary --schema cora
# imdb
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp  --getsim --typed_eval --ternary --schema imdb
# music
python -u mains_explain_.py -c -l ./experiment/music/music.lp  --getsim --typed_eval --ternary --schema music --data 50
# music-corr
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp  --getsim --typed_eval --ternary --schema music --data 50-corr
# pokemon
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp  --getsim --typed_eval --ternary --schema pokemon

```



### 2 Main Results (only after Sim facts computed)

```
# dblp
## lb
python -u mains_explain_.py -c -l ./experiment/dblp/dblp.lp   --main --lb --no_show --ternary --presimed --schema dblp
## ub
python -u mains_explain_.py -c -l ./experiment/dblp/dblp.lp   --main --ub --no_show --ternary --presimed --schema dblp
## maxsol
python -u mains_explain_.py -c -l ./experiment/dblp/dblp.lp -a -m ./experiment/maxsol-eqr.lp --main  --no_show --ternary --presimed --schema dblp
## possible merge
python -u mains_explain_.py -c -l ./experiment/dblp/dblp.lp  --pos-merge all --ternary --no_show --presimed --schema dblp


# cora
## lb
python -u mains_explain_.py -c -l ./experiment/cora/cora.lp   --main --lb --no_show --ternary --presimed --schema cora
## ub
python -u mains_explain_.py -c -l ./experiment/cora/cora.lp   --main --ub --no_show --ternary --presimed --schema cora
## maxsol
python -u mains_explain_.py -c -l ./experiment/cora/cora.lp -a -m ./experiment/maxsol-eqr.lp --main  --no_show --ternary --presimed --schema cora
## possible merge
python -u mains_explain_.py -c -l ./experiment/cora/cora.lp  --pos-merge all --ternary --no_show --presimed --schema cora


# imdb
## lb
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp   --main --lb --ternary --presimed --schema imdb --type_eval
## ub
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp   --main --ub  --ternary --presimed --schema imdb --type_eval
## maxsol
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema imdb --type_eval
## possible merge
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp  --pos-merge all --ternary --presimed --schema imdb --type_eval


# music
## lb
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --main --lb --ternary --presimed --schema music --type_eval --data 50
## ub
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --main --ub  --ternary --presimed --schema music --type_eval --data 50
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/music.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema music --type_eval  --data 50
## possible merge
python -u mains_explain_.py -c -l ./experiment/music/music.lp  --pos-merge all --ternary --presimed --schema music --type_eval --data 50


# music-corr
## lb
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp   --main --lb --ternary --presimed --schema music --type_eval --data 50-corr
## ub
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp   --main --ub  --ternary --presimed --schema music --type_eval --data 50-corr
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema music --type_eval  --data 50-corr
## possible merge
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp  --pos-merge all --ternary --presimed --schema music --type_eval --data 50-corr


# pokemon
## lb
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp   --main --lb --ternary --presimed --schema pokemon --type_eval
## ub
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp   --main --ub  --ternary --presimed --schema pokemon --type_eval
## maxsol
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema pokemon --type_eval
## possible merge
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp  --pos-merge all --ternary --presimed --schema pokemon --type_eval

```



### 3 Multi-level Recursion

```
# imdb

## ub
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp  --typed_eval --ub --presimed --ternary --rec-track --schema imdb
## maxsol
python -u mains_explain_.py -c -l ./experiment/imdb/imdb.lp -a -m ./experiment/maxsol-eqr.lp -rec-track  --ternary --presimed --schema imdb --type_eval


# music
## ub
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --rec-track  --ub  --ternary --presimed --schema music --type_eval --data 50
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/music.lp -a -m ./experiment/maxsol-eqr.lp -rec-track  --ternary --presimed --schema music --type_eval  --data 50



# music-corr
## ub
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp   --rec-track  --ub  --ternary --presimed --schema music --type_eval --data 50-corr
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/music-corr.lp -a -m ./experiment/maxsol-eqr.lp -rec-track  --ternary --presimed --schema music --type_eval  --data 50-corr


# pokemon
## ub
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp   --rec-track  --ub  --ternary --presimed --schema pokemon --type_eval
## maxsol
python -u mains_explain_.py -c -l ./experiment/pokemon/pokemon.lp -a -m ./experiment/maxsol-eqr.lp -rec-track  --ternary --presimed --schema pokemon --type_eval

```


### 4 Varying Duplicates Percentage
#### Step 1: sim computing
- For Dup30
`python -u mains_explain_.py -c -l ./experiment/music/music.lp  --getsim --typed_eval --ternary --schema music --data 30`

- For Dup10
`python -u mains_explain_.py -c -l ./experiment/music/music.lp  --getsim --typed_eval --ternary --schema music --data 10`

#### Step 2: derive solutions

```
# music 30
## lb
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --main --lb --ternary --presimed --schema music --type_eval --data 30
## ub
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --main --ub  --ternary --presimed --schema music --type_eval --data 30
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/music.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema music --type_eval  --data 30
## possible merge
python -u mains_explain_.py -c -l ./experiment/music/music.lp  --pos-merge all --ternary --presimed --schema music --type_eval --data 30



# music 10
## lb
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --main --lb --ternary --presimed --schema music --type_eval --data 10
## ub
python -u mains_explain_.py -c -l ./experiment/music/music.lp   --main --ub  --ternary --presimed --schema music --type_eval --data 10
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/music.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema music --type_eval  --data 10
## possible merge
python -u mains_explain_.py -c -l ./experiment/music/music.lp  --pos-merge all --ternary --presimed --schema music --type_eval --data 10

```



### 5 Varying Sim Thresholds
#### Step 1: sim computing
For $$\delta \in \{98,95,90,85\}$$

execute

`python -u mains_explain_.py -c -l ./experiment/music/thresh/music-{\delta}.lp  --getsim --typed_eval --ternary --schema music --data 50`



#### Step 2: derive solutions

For $$\delta \in \{98,95,90,85\}$$

execute
```
## lb
python -u mains_explain_.py -c -l ./experiment/music/thresh/music-{\delta}.lp   --main --lb --ternary --presimed --schema music --type_eval --data 50
## ub
python -u mains_explain_.py -c -l ./experiment/music/thresh/music-{\delta}.lp   --main --ub  --ternary --presimed --schema music --type_eval --data 50
## maxsol
python -u mains_explain_.py -c -l ./experiment/music/thresh/music-{\delta}.lp -a -m ./experiment/maxsol-eqr.lp --main  --ternary --presimed --schema music --type_eval  --data 50
## possible merge
python -u mains_explain_.py -c -l ./experiment/music/thresh/music-{\delta}.lp  --pos-merge all --ternary --presimed --schema music --type_eval --data 50

```



### 6 Proof Tree
#### 1) Supported Merge (-pos-merge can be replaced by any pair)
```
python -u mains_explain_.py -c -l ./experiment/music/music.lp --pos-merge rec-2489115-dup-1,rec-2489115-dup-0 --attr release,release --trace  --ternary --presimed --typed_eval --schema music --data 50
```

#### 2) Unsupported Merge (-pos-merge can be replaced by any pair)
```
python -u mains_explain_.py -c -l ./experiment/music/music.lp --pos-merge rec-2489115-dup-1,id-748010  --attr release,release --trace  --ternary --presimed --typed_eval --schema music --data 50
```

#### 3) Violated Merge (-pos-merge can be replaced by any pair)
```
python -u mains_explain_.py -c -l ./experiment/music/music.lp --pos-merge id-748086,id-748010   --attr release,release --trace  --ternary --presimed --typed_eval --schema music --data 50
```