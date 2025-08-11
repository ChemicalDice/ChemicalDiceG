Chemical Dice
=============

ChemicalDiceG is a deep learning featurizer developed using unsupervised 
learning on the ChEMBL database. It captures six distinct molecular 
representations: quantum descriptors, bioactivity profiles, language model
embeddings, graph-based features, physicochemical properties, and 
2D image-based features. ChemicalDiceG takes SMILES strings as input and 
generates comprehensive embeddings for each molecule, enabling robust and 
versatile molecular characterization for downstream cheminformatics and 
bioinformatics applications.



Install packages
----------------

To use the **ChemicalDiceG** package, you need to install it along with
its dependencies. You can install ChemicalDice and its dependencies
using the following commands:

.. code:: bash

   pip install numpy pandas tqdm rdkit 
   pip install -i https://test.pypi.org/simple/ ChemicalDiceG==0.0.1


Calculation of Embeddings
--------------------------

.. code:: python

   # SMILES column should be present in the CSV file.
   #example CSV file:
   # SMILES,other_column1,other_column2
   # CC(=O)OC1=CC=CC=C1C(=O)O,1,2
   # C1=CC=CC=C1,3,4
   # C1=CC=C(C=C1)C(=O)O,1,2
   from ChemicalDiceG import  smiles_to_embeddings
   embeddings = smiles_to_embeddings.collect_features_from_csv(
      filepath="smiles.csv",
      key = "API_KEY",  # Replace with your actual API key
   )
