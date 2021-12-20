from .backtranslation import BackTranslate
from .last_sentence_deletion import DeleteLastSentence
from .num2words import Num2Words
from .random_deletion import RandomDeletion
from .replace_named_entities import ReplaceNamedEntities
from .replace_numerical_entities import ReplaceNumericalEntities
from .replace_units import ReplaceUnits
from .same_sentence import SameSentence
from .tfidf_replacement import TF_IDF_Replacement
from .unit_expansion import UnitExpansion
from .most_important_phrase_remover import MostImportantPhraseRemover
from .negate_question import NegateQuestion
from .pegasus import Pegasus
import nltk
nltk.download('punkt')
nltk.download('wordnet')