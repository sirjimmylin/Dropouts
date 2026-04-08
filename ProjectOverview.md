# CSCI 1470 Deep Learning Final Project

The final research project is aimed to give you an idea of what a deep learning research project entails, and hopefully, get you excited about doing research in this field. It requires critical thinking that you will develop by learning the material and doing assignments during the semester. 
At the end of the semester, you will share your work with your peers through presentations and writing a report. This type of project-based exercise will help you in developing skills to perform independent research like:
- Thinking of a project idea 
- Doing literature survey
- Designing experiments
- Reporting results in a clear and concise manner. 

**Please read this handout in its entirety.** It contains all the information, forms, and **deadlines** you'll need to know about! 



## Overview

You will complete final projects in groups of **3-4 people**. These projects are an opportunity for you to apply the knowledge you've gained in class to a topic or area of interest. Your group may implement research papers, existing projects, or create entirely new projects. **The time commitment per student should be *approximately equal regardless of the group size*, thus larger groups should justify their group size by taking on more ambitious projects.**

Your project group, once assembled, will be assigned a **mentor TA**, who will guide you throughout the remainder of this semester. Your mentor will be the one to give you pointers on how to get started, where to look for ideas, and also be one of the people evaluating your work. You will be graded on completion of goals, punctuality with which you meet your deadlines, and professionalism in reports and presentations.

We do not expect you to build a magical model that solves all of the world's problems. What we do expect is concentrated and well thought out effort into solving a problem: Rather than reporting a single number using a certain metric and showing your model just "works", we expect you to perform **quantitative ablation studies** (e.g. show us what architectural changes, hyper parameters, and regularization techniques you tried and what are helpful, explain why you think so) to analyze your models, along with **qualitative evaluation** (e.g. visualizations) to illustrate what the model learns and how it might fail. You might want to check Prof. Bill Freeman's awesome talk on "How to Write a Good CVPR Paper" ([video](https://www.youtube.com/watch?v=W1zPtTt43LI&t=2681s), [slides](https://billf.mit.edu/sites/default/files/documents/cvprPapers.pdf)). Of course, we do not expect you to finish a project at "top AI conference" level within a semester, but the general high-level principles are helpful.

Do **not** be afraid of negative results! Some of the most interesting and well-received presentations from past years were ones that failed to produce "successful" results!

## Requirements

- Groups of 3 to 4 students (A solo or duo project is possible, but not recommended, for 1470 with capstone or Ph.D. students. The student must check with the instructor and get approval.)
- The project must be related to course material and must involve "training a model", meaning you can't just use a pre-existing model, like GPT5 or another closed-source LLM, to solve some problem.
- Present and submit a poster/slides of the project at the end of the semester recapping your work using GitHub. 
- We will host a "Deep Learning Day" at the end of the semester. Student groups will participate in-person for poster presentations.
- Participate in "Deep Learning Day" by engaging with other student's projects at the end of the semester.
- Meet all deadlines and check-ins specified below.
- Submit all project-related code via sharing a GitHub Repo link to your mentor TA. Projects without code submission are considered incomplete and will not be graded. Note: Even if you are implementing an existing paper, your code submission must consist of your **own original work**.
- Your code can be written using any deep learning library, such as TensorFlow, Keras, Jax, or PyTorch.

**NO** Late Days may be used for the final submission date, but if you absolutely need an extra day or two for earlier checkpoints, talk to your TA; they will approve this at their own discretion.

## Scope
The project is "open-ended," meaning it's open to interpretation. For 1470-noncapstone students, there are two options. For 1470-Capstone students, you must choose the **second** option of solving a new problem.

### Option 1: Critical Replication

Find a paper from a recent ML conference and reimplement it. The goal isn't just to get the model running, but to figure out *why* it works, whether its claims actually hold, and where it breaks.


**Your implementation must satisfy all of the following:**

1. Evaluate on dataset(s) not used in the paper
2. Extend or stress-test the implementation in a non-trivial way (explained below)

**Additionally, your project must include both of the following components:**

#### Component A: Claim Verification

Pick at least **two specific quantitative claims** from the paper and rigorously test whether they hold in your reimplementation. You're not just checking whether the model trains. It is your job to inspect whether it does what the authors say it does, under what conditions, and where it diverges.

Your final writeup should include a section along the lines of **"What the Paper Doesn't Tell You"** that addresses:
- Where your results matched the paper's claims, and where they didn't
- At least one "stress test": a distribution shift, a degenerate input, or an edge case the paper doesn't address
- Your best explanation for any divergence you observed

#### Component B: Hypothesis-Driven Ablations

Before running experiments, you must formulate and submit specific hypotheses about your model. In your **Intermediate Report**, include at least **2 written hypotheses** about which components of the architecture are most critical and why. You will discuss these with your TAs during project check-in #2!

Your final writeup's ablation section will be evaluated on:
- Whether your experiments actually test your stated hypotheses
- How honestly you engage with results that went against your predictions
- The quality of your explanation for *why* components matter, not just *whether* they do

:::warning
**On AI tool use:** You may use AI tools for coding assistance, but the hypothesis and claim-verification sections require you to demonstrate understanding that can't be generated after the fact. Be specific. Vague hypotheses ("we think the attention mechanism matters") will not receive credit.
:::

> **What makes a good paper choice for this option?** Look for papers where the authors make strong architectural claims ("component X is essential because...") or where results seem surprisingly good on a benchmark. Those are the papers where verification is most interesting and most likely to turn up something worth writing about.

---
### Option 2: Try to solve a new problem

You can do this using whatever deep learning methods you can find that get the job done. Ideally, the project would involve more than one major topic we covered in the class (CNNs, RNNs or Transformers, Generative models, Fairness and model interpretability, and Reinforcement Learning). For 1470-capstone students, your capstone requires you to work on a project that connects what you have learned in more than one course (e.g. deep learning, machine learning, and computer vision). You are encouraged to implement your project "from scratch", but can also use open-source deep learning projects as a component in your framework. Example use of open-source projects that are permitted:

- You are building a generative model that turns food videos into recipes, you can extract visual features with an open-sourced model, such as [CLIP](https://github.com/openai/CLIP).
- You want to build a reinforcement learning model for a game you invented, you can try out some well implemented [RL baselines](https://github.com/openai/baselines).
- You run a thorough analysis of model bias (e.g. gender, racial) for one or a collection of popular deep learning models that have been open-sourced.

Example uses that are **not** permitted:

- You take open-sourced model checkpoints and just "fine-tune" them on another dataset.
- You take an open-sourced framework and replace its ResNet-50 with a ResNet-101.

Please cite all the open-source frameworks you used in the final report, and check with your mentor TA if they are okay with your proposal.

You can check out the projects in the [previous Deep Learning Day](https://brown-deep-learning-day-f2021.devpost.com/project-gallery) or [Stanford CS231n](http://cs231n.stanford.edu/2017/reports.html) to draw some inspiration, but your project needs to differentiate from previous projects in meaningful ways.
## Gen-AI Use

We are allowing the use of AI tools (such as ChatGPT, Claude, Copilot, etc.) for coding assistance in the final project. However, you (the student) are responsible for understanding all content that you submit and present. AI tools can help with debugging, syntax, and implementation suggestions, but you must be able to explain how your code works, why you made specific design choices, and the details of your model architecture. You will be asked about implementation details during check-ins and presentations. Saying "we used ChatGPT for that" is not a valid explanation for how you implemented something. 

## Compute Resources

We recommend a project that can run on your desktop, department machines, or on Oscar in a reasonable amount of time. Please check with your TAs if your project proposal might require excessive compute resources than we can provide. Remember that you will not be judged for the absolute performance (e.g. you get the best numbers in the world on some benchmark), but the creativity of ideas, quality of code and documents, as well as thoroughness of the ablation experiments.

If you really want to take on a very ambitious project that requires several GPUs with large memories, please ensure that you have obtained the computational resources **before** you start with the project and get the TA's permission. Example resources:

- You are working in a research lab and the lab provides you with GPU machines.
- Try Cloud service providers that offer free student credits (e.g. GCP). Google colab notebooks will also let you use 1 GPU for free. 

## Deliverables

### Forming teams
**Due:** Wednesday, March 18

Please fill out this [Google form](https://forms.gle/gy7TkTczWvMYeUnK6) to let us know if you would like to form your own team or would like us to assign you a team. **If you have decided to form your own team,** please also submit the names of your team members and one form submission per team is sufficient. Remember you can form a group of 3-4 people.

**Final team assignments will be shared by Friday, March 20th**

### Project Check-in #1

**Due:** Week of March 30th

For your project check-in, you (with your team members) will meet with your mentor TA and have a brainstorming session. Reach out to your mentor TAs to set up a meeting during the week of 03/30 (until 04/04). You should prepare a few ideas in advance to discuss the plausibility as well as scope of the project. This could include some application domains that you are interested in, a paper or 2 that you found interesting, some deep learning model you really want to implement, etc. This check-in is your opportunity to start thinking about your project proposal as a team and get some guidance for the same from your TA mentor.

### Project Proposal

**Due:** April 3, 11:59pm

After meeting with your mentor TA, you will then submit a project proposal. With your team members, decide a team name, and submit your final project idea by filling out the form [here](https://forms.gle/2m5uBepA4Htz9Zrg8)! Only one person from your group needs to submit the form for everyone. If you are re-implementing an existing paper, please cite the paper that you want to implement. If you are trying to solve something new, please describe the problem and your plan of action. We will approve all that are appropriate.

Please note that if you do not submit your proposals by the deadline, you will receive a **2% deduction** on grade for this project. This deadline cannot be extended except for in extenuating circumstances, as TAs need to review and approve your proposals before greenlighting your project before you begin working on it.

___



### Project Check-in #2

**Due:** Week of April 19th
[**Rubric**: TAs will be grading this check-in based on this rubric.]()

For this check in, you will 1) write a one-page reflection on your progress so far and 2) meet with your mentor TA. We expect you are wrapping up the implementation and beginning to perform experiments. If you have questions before the check-in, please contact your mentor TA, or post questions on Ed.

**Submit the reflection (as described below) by posting it on your repo or emailing your mentor TA before your meeting.**

For this checkin, we also require you to write up a reflection including the following:
- **Introduction**: This can be copied from the proposal.
- **Challenges**: What has been the hardest part of the project you've encountered so far?
- **Insights**: Are there any concrete results you can show at this point? 
    - How is your model performing compared with expectations?
- **Plan**: Are you on track with your project?
    - What do you need to dedicate more time to?
    - What are you thinking of changing, if anything?

This check in meeting with your mentor TA can either be in-person or over Zoom, Google Meet, etc. Reach out to your mentor TA to schedule this meeting.

Regarding what we generally expect you to have **done** by this time:
- You should have collected any data and preprocessed it. 
- You should have shared the GitHub repo link with your mentor TA 
- You should have almost finished implementing your model, and are working on training your models and ablation experiments.
- Please make sure you are keeping your list of public implementations you've found up-to-date. 


### Deep Learning Days)


- **Dates:** April 29th, April 30th, and May 1st of 2026
- **Time:** 3 Sessions
    - 10AM - 12PM
- **Location:** Sayles Auditorium and Kasper 

:::danger
If you have any conflict with the timing or date of Deep Learning Day, please fill out this form as soon as possible so we can figure something out!
:::

This is a chance to show off your team's awesome project and see all of the great work your peers have done! You'll be expected to attend your theme session to present your work and to ask question to other groups. More logistical details about the event and participation will be shared as we get closer to the event. 

1470 groups should be prepared to give a ~2 minute presentation of their poster.

Your poster must contain the following information:
- **Title**
- **Names of project group members**
- **Introduction**: what problem you're solving and why it's important
- **Methodology**: your dataset and model architecture, etc.
- **Results**: both qualitative *and* quantitative (e.g. if you're doing an image-related project, we want to see both pictures *and* graphs/tables) 
- **Discussion**: lessons learned, lingering problems/limitations with your implementation, future work (i.e. how you, or someone else, might build on what you've done)


### Final Projects Due

**Due:** 11:50 PM, May 3rd, 2026

You will need **three final deliverables** the due date (note this is a **hard deadline**):
1. Poster - All students will post their digital posters on their GitHub repo
2. Finalized code on GitHub
3. Final writeup/reflection

#### Poster
For poster presentations, we require one high resolution horizontal 4:3 poster (in the form of a JPG) to be displayed on your GitHub repo. You should keep things sufficient and also make it visually appealing.  We recommend using InDesign, PowerPoint, or LaTex for your poster. 

#### Final Writeup/Reflection
Along with your GitHub Repo, provide a final write up/reflection of the final project. 

The goal of this Final Writeup is to act as your "research paper". Therefore, we want it to reflect your thought process and walk us through the steps of developing your project in an academic context. This is some high-quality and high-effort work you have done, so why not wrap it up in a nice pretty bow (paper) that explains all of your work.

:::success
For the best formatting, we **highly** reccomend using LaTeX. Specifically, we have linked a template for you to use if you choose to. This template will open Overleaf (which you have access to through your Brown login).

{%preview https://www.overleaf.com/read/bsjnjggxsvfr#2efef3 %}

:::warning
We have defined some helpful "macros" which make writing a lot of the tedious formatting easier. For example, instead of needing to write`\mathbf{x}` each time, you can just write `\x`. We have defined a handful of these at the top as well as some examples throughout, so take a look and feel free to use any.
:::

Here are the following sections you should be sure to include:
- **Title**
- **Who**
- **Introduction**
- **Literature Review**
- **Methodology**
- **Results**
- **Challenges**
- **Ethics**
- **Reflection**

We reccomend the final report be ~5-7 pages long. Too much over and you likely are being overly verbose for this kind of report. Too far under and you won't sufficiently explain your choices.

Note that most of this writeup shouldn't require much extra effort. This is just the culmination of all of your work that you have been slowly chipping away at over the past few weeks. 

You can use your previous submissions in your final report. For example, the "Introduction" and "Methods" should be mostly adaptable from your initial outline (although be sure to modify accordingly if you pivoted or otherwise adjusted from the initial outline and maybe add in some fun prose), and "Challenges" can build off of what you discussed in your checkpoint #2 reflection. The "Results" section can summarize your results as they are in the poster; this is also a space to add any additional results that didn't make your poster. In your final reflection, please address the following questions (along with other thoughts you have about the project, if any):
- How do you feel your project ultimately turned out? How did you do relative to your base/target/stretch goals?
- Did your model work out the way you expected it to? 
- How did your approach change over time? What kind of pivots did you make, if any? Would you have done differently if you could do your project over again?
- What do you think you can further improve on if you had more time?
- What are your biggest takeaways from this project/what did you learn? 


## Grade Breakdown
A project submission is considered complete if the written report, presentation, **AND** code are all submitted. An incomplete project receives zero grade. A complete project will be graded as:
- Written reports: 45%
- Poster + Oral Presentation: 35%
- Code: 15%
- Peer evaluation: 5%
    - To encourage a fair distribution of work between group members, each student will fill out this [form](https://forms.gle/TrZEJ7E7H6mjqFB38) at the end of the semester in which they describe the contributions that every other group member made to the project.

## Previous Final Project Posters/Reports
To help with knowing what's expected of you for your final project posters and reports, here are some DevPost links of DL Days from past offerings (you might have to log in to see the projects):
- [Spring 2023](https://brown-deep-learning-day-s23.devpost.com/)
- [Fall 2022](https://brown-deep-learning-day-f2022.devpost.com/)

We encourage you to also check out some of the projects that current and past TAs implemented in previous years:
- What the f&nt? ([poster](https://drive.google.com/file/d/1jRNYz6BC6JnPVeBxtEyur8Snro_GlECI/view), [writeup](https://docs.google.com/document/d/1G1yaYVTVwvIwT_JxketJEGNGAMkp2BuI5DUtAEiMVeM/edit?usp=sharing))
- Social Media Fake News Detector ([poster](https://docs.google.com/presentation/d/1KwmZb1-IOV0GSeQgFtu1AeXlNpphmkWx6udelo60KLc/edit?usp=sharing), [writeup](https://docs.google.com/document/d/1Dt8ir_jogCRbEcb9aWegED3CayN4NmBlH4ufNvGbDVA/edit?usp=sharing))
- Computational Photography in Extreme Low Light ([poster](https://drive.google.com/file/d/1Xh7U88LvsjTeSgaMy87ppBxmMBSYUudS/view?usp=sharing))
- Colorizer ([poster](https://docs.google.com/presentation/d/1r8hzBGwNVbMGL55a7pP0Hi_MkwZd5hjG6Z2wSg9YOZU/edit), [writeup](https://docs.google.com/document/d/1uWwGWh3g_jzh4wdPc5KjIeos-bYt2A6XXN4p90ZjJk4/edit))