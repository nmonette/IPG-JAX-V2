# IPG-JAX-V2
A JAX implementation of the Paper "Learning Equilibria in Adversarial Team Markov Games: A Nonconvex-Hidden-Concave Min-Max Optimization Problem" by Jingming Yan, Fivos Kalogiannis, and Ioannis Panageas.

To run the experiments:
- Install the docker image:
    ```
    cd setup
    chmod +x build.sh
    ./build.sh
    cd ..
    ```
- Run the docker: (replace `GPU_IDS` with the real GPU id values)
    ```
    docker run -it --rm --gpus '"device=GPU_IDS"' -v $(pwd):/home/duser/ipg-jax ipg-jax
    ```
- Run `train.py` with whichever args you desire. The args are self explanatory and are outlined in the `Args` class in [train.py](train.py)
    - There are two environments installed currently: 
        - Matrix:
            - The configuration of the matrix environment can be found and customized in [environments/matrix.py](environments/matrix.py)
            - It is an Adversarial Team Markov Game with a randomly generated payoff matrices sampled uniformly between $-1$ and $1$, and with a randomly sampled transition function
        - Multigrid(2):
            - The configuration of the matrix environment can be found and customized in [environments/multigrid2.py](environments/multigrid2.py)
            - Grid world
            - The team wishes to get all of the goal squares before the adversary can get any of them
