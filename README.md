To run it:

Install Requirements: pip install -r requirements.txt and install pyunicorn and GraphEmbedding.
Or clone the GraphEmbedding repository:
git clone https://github.com/shenweichen/GraphEmbedding.git
cd GraphEmbedding
python setup.py install

Run dataset.py: This will download data (if configured) and create the train/validation/test DataFrames.
Run price_graph.py: This will generate the visibility graphs.
Run price_ci.py: This will calculate the collective influence.
Run price_embedding.py: This will generate the Struc2Vec embeddings.
Run trainer.py: This will train (or test) the model.
Remember to adjust the parameters in config.yaml as needed for your specific data and experiment setup.