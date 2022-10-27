#   conda environment helpers

conda activate helpers
pip3 install -r requirements

# install pytorch with gpu support

# conda -n create env_name python=3.9
# conda activate env_name
# conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install pandas matplotlib
# conda install -c conda-forge opencv
# conda install -c anaconda pillow
# conda install -y jupyter
# conda install scikit-learn
# conda install scikit-learn-intelexy


# windows
'''
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

python
import torch
torch.cuda.is_available()
'''
