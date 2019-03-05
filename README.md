# CutSolver

This is used to solve the "Cutting Stock Problem", which is NP-hard.  
It can be reduced to the Bin-Packing-Problem (BPP).

This solver uses brute force (best solution) for small n and a heuristic (fast solution) für larger n.

# Usage
Make sure that you have installed Docker.  

1. Build this image using `docker_build.sh`
1. Start using `docker_start.sh`
1. Send POST-Requests to `[localhost]/solve`, see `/docs` for further informations.

This Solver is using ints exclusively, as there is no need for arbitrary precision yet. 
It also has no concept of units so you can use whatever you want.

# Visualisation

![cutsolver](https://user-images.githubusercontent.com/25404728/53304884-fb9c4980-387a-11e9-9a49-330369befc44.png)

### Declined
Having workers and a queue with pending jobs was considered but seemed useless, 
as ideally all requests have their own thread and a (by comparison) short calculation time.
This makes a queue useless. The same argumentation also holds for a result-buffer.

# Dependencies
*Everything should be handled by Docker*

This project uses:
1. [pipenv](https://github.com/pypa/pipenv): library management
1. [FastAPI](https://github.com/tiangolo/fastapi): easy webservice (this includes much more!)

# External links
https://scipbook.readthedocs.io/en/latest/bpp.html


