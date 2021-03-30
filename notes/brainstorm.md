# Brainstorm

* Use graph from fastepidem
* Package structure
* Separate modules for each model (Class + its "outside" numbed functions)
* Use the "smart allocation" structure from fastepidem.
* Parallelize using numba!?? So I wouldn't have to create different python instances.
* Can I separate single layer and multiplex implementations?? Hmmm, seems overly complicated. Better do it all multiplex.
* Graph/multiplex and model are separate data bunches. One interacts with the other, but _no parenting relationship_. It's an _association_ relationship.
* names for the transitions

# Data containers

## Graph layer and multiplex

### Layer

* ``neighbors_list`` , ``neighbors_alist``, ``neighbors_awk`` - adjacency structure,  respectively as list of lists, list of arrays and awkward array. It is a ragged 2-level structure. Signature: 
  * ``neighbors_alist[i, j]`` = j-th neighbor of node i
  * ``neighbors_list[i][j]``
  * ``neighbors_awk[i, j]``

### Multiplex

Data bunch that simply collects layers in an associative relationship. No creation/destruction, just a check for equal sizes.



## Model



### ModelBase

Holds placeholders for generic attributes of a node, such as its states, its "prevalences", 

### Model implementations 

* ``p_state_s[i]``  - Used as a state-alias of p_state[], for readability inside model methods.
* ``p_next_s``
* ``f_trans[i_tr, i]`` a node's ability to not-promote a given transition on its neighbors
  * = transition indexed by i_tr, node i
* ``f_trans_tr[i]`` = alias for a line of ``f_trans``
* ``q_trans``: a node's propensity to undergo a given transition

**Alternatively** : use 2D arrays for the states (by index), then use aliases for each line to improve legibility

* ``p_state``: 2D array, where each line represents a state and columns are nodes.
  * `p_state[i_s, i] ` = probability that node *i* is on state of index *s_i* 

# Functions and Methods

* ``calc_f_trans``

* ``calc_q_trans``

  * ### Dumb order (get value from neighbors):

  * For each node i 

    * For each neighbor j of i 
      * multiply all f-factors of neighbors

  * ### Interesting order (broadcast value to neighbors)

  * initialize all q-factors as 1

  * for each node i:

    * f = calc_f_trans_tr[i]
    * for each neighbor j of i
      * multiply q_trans[j] by f

  * Set q = 1. - q

  