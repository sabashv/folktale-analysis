## Paper: Comparative phylogenetic analyses uncover the ancient roots of Indo-European folktales (2016)

### Question: 
Do folktales spread mainly by: inheritance from ancestral cultures (vertical transmission), or borrowing between neighboring societies (horizontal transmission)?

### Data: 
- 275 folktales (ATU = “Tales of Magic”), Tales were coded as present or absent in each population 
- Indo-European language family tree
- Geographic distances between populations

### Method: 
Each folktales is considered one cultural trait. Compare train distribution with random distribution to see if they follow the language tree (phylogenetic signal) or the geographical distances. (phylogenetic signal tests, autologistic regression, ancestral state reconstruction)

### Findings:
- Many folktales follow language ancestry more than geography
- Spatial proximity sometimes has a negative effect: suggesting that societies were more likely to reject than adopt these stories from their neighbours.
- This raises a more general question about why populations seem to readily adopt some tales from their neighbours, while apparently rejecting others. Theoretical studies of cultural evolution suggest that patterns of cultural diversity are often shaped by parochial transmission biases (e.g. conformism, neophobia) that inhibit the exchange of information between groups and preserve local distinctions. However, relatively little work has examined the extent to which these biases target particular kinds of traits, or the circumstances under which they might be relaxed. 

---

## Paper: Cross-Cultural Analysis of Human Values, Morals, and Biases in Folk Tales (EMNLP-2023)

### Data: 
- 1,900 folk tales (From Ashliman, Multilingual Folk Tale Database, and Project Gutenberg).
- These stories originate from 27 diverse cultures(countries) across six continents.

### Method: They use lexicon-based and correlation analyses to measure:
- Values: First, we tokenize and lowercase each tale with spaCy. We then compute a distribution of human values for each tale by counting tokens associated with a human value and normalizing by the total number of tokens in the text. The 49 values from the Values Lexicon: Social, Siblings, ...
- Moral foundations: a lexicon-based analysis of moral foundations by employing the Moral Foundations Dictionary, which associates 2,103 words and phrases with the 10 moral foundations in MFT.
- Gender bias: assign each story a dominant gender focus, then analyse gender–value associations across cultures

===

## Use case 1: Motif-level phylogenetic signal.
- Which motifs are vertically transmitted which ones are horizentally transmitted?
- Which kinds of motifs are rejected by neighbors, and which are shared?

## Use case 2: motifs to track how moral norms, gender roles, and values change across regions

---
(we need to add timestamp to the data)
## Use case 3: Which motifs survive across many ATU types and cultures, and which die out?
## Use case 4: motifs to track how moral norms, gender roles, and values change across time

