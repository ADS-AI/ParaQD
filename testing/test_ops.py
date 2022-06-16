import os, sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.operations import ReplaceNamedEntities, ReplaceNumericalEntities, TF_IDF_Replacement

sent = "John ate 5 mangoes, while Steve ate 5 kgs of apples."

print(TF_IDF_Replacement().generate(sent))