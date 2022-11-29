---
title: "Generating Vice Headlines using Bloom"
date: 2022-10-14T00:00:00+00:00
image : "images/Bloom.jpeg"
# Img source: 
# author
author : ["Admin"]
# categories
categories: ["NLP", "Project"]
tags: ["NLP","Hugging Face"]
# meta description
description: "How to Generate YouTube Titles using BLOOM"
# save as draft
draft: false
---

## Motivation


<a title="Unknown authorUnknown author, Public domain, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Vice_Logo.svg"><img width="256" alt="Vice Logo" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Vice_Logo.svg/256px-Vice_Logo.svg.png" style="float: right;margin: 0 0 0 15px;"></a>
> *“What It’s Like To Sell Burgers In North Korea”*  
  
About three years ago, this title sent me down yet another rabbit hole of obscure videos on VICE News. And how could it not? North Korean life is already quite opaque, but who knew they sold burgers there?

Aside from the oddball videos, such as *“A Male Witch Explains Why He’s not a Wizard”* and *“I’m the Victim of a Far Right Conspiracy Theory”*, the channel features lots of reporting on conflict regions and events happening on the fringes of society. I started to wonder…, could a machine pick up on this? Would a language model be able not to learn, but replicate this subtle weirdness inherent to VICE YouTube Titles? I wanted to know, and already knew how to find out: fine-tuning BLOOM!




#### BLOOM

A few months ago, BigScience made waves in the open-source community by releasing their new large language model [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom) to the public. It contrasted itself from most previous models by not making the number of parameters or dimensions its main selling point; it was its’ availability. One of BigScience's main competitors, OpenAI, hesitated to make their new model GPT3 publicly available. Only after months of exclusive commercial licensing to Microsoft was the average person able to access it. OpenAI cited their commitment to the safe deployment of AI as grounds for this decision, but critics labelled this a [monetisation issue](https://www.technologyreview.com/2020/09/23/1008729/openai-is-giving-microsoft-exclusive-access-to-its-gpt-3-language-model/). In contrast, BLOOM was open-sourced from the get-go, ready to use by anyone with enough technical savvy to run a [Hugging Face](https://huggingface.co/) model.

## How To Build It

### Data

> [<em>Data is at the heart of every ML endeavour.</em>](https://towardsdatascience.com/from-model-centric-to-data-centric-artificial-intelligence-77e423f3f593)


To my knowledge, there exists no readily available dataset of VICE video titles. We have to create one ourselves. This will be fairly easy, with YouTube providing a proprietary API to query data.

#### Gathering Data

There are a few setup steps that we need to complete before using the API. Roughly speaking, they are:

1. Create a [Google Developer Account](https://developers.google.com/)
2. Create a new project
3. Enable the [YouTube Data API v3](https://developers.google.com/youtube/v3)
4. [Create a credential](https://developers.google.com/youtube/registering_an_application)
5. Add the credential to your workspace

There is an [excellent Hubspot article](https://blog.hubspot.com/website/how-to-get-youtube-api-key) to guide you through the setup.

Once you have the YouTube API working, we can use it to query the titles of all videos of a channel. There is a Python [script](https://github.com/marcderbauer/bloom/blob/82c2cd11fb93477b08e4beff80474e682e16f570/youtube/query_api.py) included for this in this project’s GitHub repo.

All the script needs is a YouTube playlist ID. We can find the playlist ID of an entire channel by going to the channel's page, clicking `PLAY ALL` and copying everything after `list=` in the link.

```latex
https://www.youtube.com/watch?v=Q6wBh7UcHqo&list=n8zNIfYAQNdrFRrr8oibKw              # VICE
https://www.youtube.com/watch?v=cDppHAzyFew&list=PLw613M86o5o7q1cjb26MfCgdxJtshvRZ-    # VICE News

```

To maximise the amount of training data, it makes sense to include both the VICE and VICE News channels. They are topically similar, and for our intents identical.

The `query_api.py` makes saving video titles easy. We just need to feed the playlist IDs to the script like this:

```latex
python3 youtube/query_api.py UUn8zNIfYAQNdrFRrr8oibKw PLw613M86o5o7q1cjb26MfCgdxJtshvRZ-
```

It automatically downloads all video titles of a given playlist ID and concatenates them into one document named after the channel. Given multiple playlist IDs, it also creates a combined.txt containing all titles. This is the file we will use.

#### Cleaning data

All the data we just queried is of a single type: YouTube Titles. They are already quite regular, but we can do better. Since we want to train a machine learning model, should aim for the highest data quality possible. We still have some more cleaning to do!

On first inspection, there are three main issues: 

1. Videos being in foreign languages i.e. non-English
    
    ```md
    Dibattito: la settimana lavorativa da 4 giorni in Italia è possibile?
    ```
    
2. Video series, creating multiple entries for essentially the same title
    
    ```md
    North Korean Labor Camps (Part 1 of 7)
    North Korean Labor Camps (Part 2 of 7)
    North Korean Labor Camps (Part 3 of 7)
    ...
    ```
    
3. Tags attached to categorise the videos
    
    ```python
    ANTHONY PAPPALARDO PART 1 of 2 | EPICLY LATER'D | VICE
    VICE Guide to Karachi with Suroosh Alvi : Riding with a Killer (Part 5/5)
    ```
    

The first two issues are easy to fix programmatically. Using the [langdetect](https://pypi.org/project/langdetect/) library, We can filter out foreign-language videos from the list as such

```python
from langdetect import detec

def filter_language(lines, language="en"):
    return list(filter(lambda x: detect(x) == language, lines))
```

To filter similar video titles, we use the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance). It’s a simple similarity measure to capture the difference in characters between two inputs. Using our example from above we notice that only one character is different. The Levenshtein distance for this pair is therefore on

```markdown
North Korean Labor Camps (Part 1 of 7)
North Korean Labor Camps (Part 2 of 7)
```

In Python, we use of the [Levenshtein library](https://pypi.org/project/python-Levenshtein/).

```python
from Levenshtein import distance

def filter_similar(lines, max_distance=3):
    seen_lines = []
    for line in lines:
        seen = False
        for seen_line in seen_lines:
            if distance(line, seen_line) <= max_distance:
                seen = True
                break
        if not seen:
            seen_lines.append(line)

    return seen_lines
```

The two filters remove 1138 of the 7801 original titles, leaving us with a training set of 6663 examples.

##### Manual Corrections:

In the previous steps, we filtered out unwanted lines, but have yet to touch any of the contents within the lines. 

Many titles have tags to associate videos with ongoing series. The tags are hard to remove programmatically and vary on a channel-by-channel basis. Furthermore they often don’t follow strict patterns. Some are separated by a pipe “|”, some by a dash “-”, or even by being surrounded by brackets ”(…)”. To add onto the complexity, tags aren’t always suffixes. Some tags are prefixes. This means that you can’t just remove everything occurring after some marker, since it may erase the main content.

```md
# Suffix
Did This YouTuber Actually Crash A Plane for Views? | My Life Online
Documenting America's Underbelly - ALL GAS NO BRAKES
How to Eat Like a Northerner (MUNCHIES GUIDE To...Full Episode)

# Prefix
The Opioid Effect: Maine's Fishing Community Battles with Heroin
The Opioid Effect: An Ohio Family Rebuilds After Addiction
```

These tags require judgement calls on a case-to-case basis. They need to be manually cleaned using RegEx. It’s important to note that the tags can be essential to the structure of the title. We, therefore, do not want to remove all of them! Ideally, we want to delete just enough to simplify training and prevent overfitting. We also need to emphasise removing suffixes, as they reduce the output space and may indirectly serve as premature `<EOS>` tokens.

### BLOOM

#### Training

Using BLOOM's 176 Billion parameter base model would be a dream come true. Unfortunately, training the model locally limits the resources available (and training the model using a cloud service is out of scope for this project!). We thus need to restrict ourselves to the 560 Million parameter version -- more than enough for a fun project!

The first step on the model side is tokenisation. BigScience provides a pre-trained `Tokenizer`, which should provide optimal results for BLOOM.

```python
# Load pretrained tokeniser
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

# Define tokenise function
def tokenize_function(examples):
    examples["text"] = [example + " </s>" for example in examples["text"]] # Append EOS
    tokenized = tokenizer(examples["text"], padding=True, pad_to_multiple_of=8)
		return tokenized

# map function onto dataset in batches
tokenized = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
```

In the `tokenize_function` we append a `<EOS>` tag, so the model will learn when to stop generating. All the tokenised examples are also padded to a multiple of 8, in this case to a length of 32. This allows for batching; training the model significantly faster with only a slight trade-off in results.

As all our padded examples are of length 32, we set a block size of 128, setting 4 parallel workers to process our examples at the same time.

```python
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4,
)
```

Now that the data is fully tokenised and batched, we can start the fine-tuning.

Fine-tuning is a technique of *transfer learning*, adjusting a model trained on one task to function on another. As the name suggests, it *transfers* the knowledge of a Large Language Model, such as BLOOM, onto a highly specialised task, such as generating YouTube Titles. 

Hugging Face provides a <code>Trainer</code> class, which only needs the model, the dataset and the desired hyperparameters for fine-tuning. This makes the process fairly straightforward. 

Our task is to fine-tune the model, not to train it from scratch. We thus opt for a small `learning_rate` of 0.00002. To prevent overfitting we add regularisation with `weight_decay=0.01`. The 10 epochs are somewhat arbitrary, as we logged the training in Weights and Biases and can just choose the best epoch according to the train and evaluation loss.

```python
# Import the model
model = BloomForCausalLM.from_pretrained(bloom-560m)

# Define Hyperparameters
training_args = TrainingArguments(
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=10
)

# Instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"]
)

# Start the training
trainer.train()
```

#### Inference

The final (and most fun) part is inference. There are [many different inference strategies](https://huggingface.co/blog/how-to-generate) we could use! The current state-of-the-art is sampling-based, so we will choose this one. It is also non-deterministic, which makes it a great choice if you don’t want your model to repeatedly generate the same sentence.

We can pass different parameters to the `model.generate` function to finely adjust inference. There are too many parameters to elaborate upon, so I will leave you with this amazing [Hugging Face article](https://huggingface.co/blog/how-to-generate) and the [official documentation](https://huggingface.co/transformers/v2.9.1/main_classes/model.html#transformers.PreTrainedModel.generate).

```python
max_length=30, 
min_length=len(input)+1,
top_p=0.92, 
top_k=50,
temperature=0.7,
repetition_penalty=1.2
```

##### Titlecase

Our final result should be a title. This means, it should follow a specific capitalisation pattern: all words start in uppercase, save for a select few function words. The `titlecase` library capitalises strings according to this pattern, so all we need to do is pass the results from the `model.generate` function to `titlecase`.

Here’s some examples our model generated:

```markdown
Inside the Secret Helpline for People Living with Mental Health
North Korean Conspiracy Theorist Takes Me on a Virtual Tour of London
Why Do So Many People Think Donald Trump Is the Answer to Social Security?
```

### Summary

Fine-tuning BLOOM is a fun exercise in applied machine learning. I am a big fan of building entire projects from start to finish; ending up with a fully functioning model is therefore extremely satisfying. Aside from being a good exercise in data sourcing/cleaning and engineering, it teaches many best practices in machine learning.

A firm believer in open-source software, I wanted to make this model as easily accessible as possible. The code for this project is available on my GitHub repository, with full documentation. You can download the model from the Huggingface Hub and try it out online with a Gradio demo hosted on Huggingface Spaces.

If you have any thoughts, suggestions or improvements, I would be happy to hear from you!

<td nowrap><>  

[![GitHub](https://img.shields.io/badge/-Github-000?style=flat&logo=Github&logoColor=white)](https://github.com/marcderbauer)
[![Model](https://img.shields.io/badge/-Model-yellow?style=flat)](https://huggingface.co/marcderbauer/vice-headlines)
[![Demo](https://img.shields.io/badge/-Demo-green?style=flat)](https://huggingface.co/spaces/marcderbauer/vice-headlines )
[![Gmail](https://img.shields.io/badge/-Email-c14438?style=flat&logo=Gmail&logoColor=white)](mailto:hello@marcanthonybauer.com?subject=[BLOOM]%20)
</td>

<a href="https://www.technologyreview.com/2022/07/12/1055817/inside-a-radical-new-project-to-democratize-ai/">Cover Image Source</a>  
