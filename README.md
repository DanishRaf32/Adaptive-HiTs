# Adaptive-Hierarchical Time-Stepping Scheme (AHiTS)
## Solve multiscale ODEs and PDEs efficiently using machine learning 
Asif Hamid, Danish Rafiq, Shahkar A. Nahvi and M. A. Bazaz

This repository is to help the users reproduce the results presented in *"Hierarchical deep learning based adaptive time-stepping of multiscale systems", Eng Appl Artif Intell, vol. 133 part D, 2024*

![ahits_block](https://user-images.githubusercontent.com/81804223/218659948-ccecf0d8-1cc2-415b-9a81-6d074d5d1506.png)


## Getting Started
1. Clone the entire directory
```
git clone https://github.com/DanishRaf32/Adaptive-HiTS.git
conda create -n <ENV_NAME> python=3.7
conda activate <ENV_NAME>
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```
To allow tqdm (the progress bar library) to run in a notebook, you also need:
```
conda install -c conda-forge ipywidgets
```
2. Generate data for all benchmark systems by running the script *"scripts/data_generation.ipynb"* (for the KS system, the data is already available)
3. Train models for all benchmark systems via script *"scritps/model_training.ipynb"* (I have already provided the models for quick run, but new models can also be trained)
4. Finally, run the file  *"scripts/adaptive-HiTs.ipynb"*


> For any technical issues, feel free to contact danishrafiq32@gmail.com

## References
<a id="1">[1]</a> 
Yuying Liu, Nathan Kutz, and Steven Brunton, Hierarchical deep learning of multiscale differential equation time-steppers
Phil. Trans. R. Soc. A., (380), 2022.
