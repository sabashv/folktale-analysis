## The exact prompts used to extract motifs/units from folktales are listed below. 
- The extraction is done using llama-3.1-8b-instant model. 
- Prompt3 (aft_df_with_extracted_motifs3.pkl) has the highest accruacy on atu classification.

#### Prompt2:

```
You are a computational folklorist analyzing narrative patterns in folktales.

Extract the key motifs from the following story.

A motif is a recurring, distinctive narrative element or pattern in a story. It can be a specific action, object, character type, supernatural event, or situational feature that appears across multiple tales and carries recognizable symbolic or functional meaning (e.g., "a forbidden door", "transformation by eating a magical fruit", "helpful animal companion", "quest for a lost object").

Instructions:
- Focus on concrete, specific elements that are clearly present in the text.
- Keep descriptions precise: use specific roles, objects, beings, or actions (e.g., "stepmother forces girl to do impossible tasks" instead of "evil antagonist", "golden apple" instead of "magic fruit").
- Do NOT add interpretations, symbolism, or elements not directly stated.
- Include culturally relevant details when they appear (e.g., kinship terms, specific titles, supernatural beings).
- Each motif should be one short, clear sentence in simple declarative form.

FORMAT:
Return a JSON object with the following structure:
{{
  "Motifs": [
    "motif1",
    "motif2",
    "..."
  ]
}}

STORY:
{story_text}
```

#### Prompt3:

```
You are a computational folklorist analyzing narrative patterns in folktales.

Extract the key recurring narrative motifs from the following story.

A motif is a recurring, distinctive narrative element or pattern in a story. It can be a specific action, object, character type, supernatural event, or situational feature that appears across multiple tales and carries recognizable symbolic or functional meaning (e.g., "a forbidden door", "transformation by eating a magical fruit", "helpful animal companion", "quest for a lost object").

FORMAT:
Return a JSON object with the following structure:
{{
  "Motifs": [
    "motif1",
    "motif2",
    "..."
  ]
}}
STORY:
{story_text}
```

#### Prompt5:

```
Given the following folktale, extract key motifs as a list of short, general phrases
(e.g., magical helper aids, journey undertaken, reward earned). Avoid using
synonyms with backslashes (e.g., scarcity strikes, not famine/scarcity). Add slight
detail for clarity, but keep phrases general and applicable across stories, avoiding
specific names or objects.
FORMAT: Return a JSON object with the following structure: {{ "Motifs": [ "motif1", "motif2", "..." ] }}

STORY: {story_text} 
```

#### Units:

```
You are analyzing a folktale to extract concrete narrative units.

TASK:
Identify recurring narrative situations in the story.

IMPORTANT CONSTRAINTS:
- Each unit must describe a single, minimal action or situation.
- Do NOT fuse multiple events into one sentence.
- Do NOT use abstract concepts.
- Avoid excessive specificity. Units should be generalizable to similar stories.
- Each unit must explicitly mention who does what or what happens, without explaining or interpreting.
- Use short sentences (one situation per unit).
- Focus on extracting 5-15 most important narrative units that drive the main sequence of events.
- Include essential context or setup.
- If consecutive actions always occur together, extract only the smallest reusable situation.
- Avoid repeating the same action multiple times in slightly different wording.

FORMAT:
Return a JSON object with the following structure:
{{
  "NarrativeUnits": [
    "short concrete description1",
    "short concrete description2",
    "..."
  ]
}}

STORY:
{story_text}
```

