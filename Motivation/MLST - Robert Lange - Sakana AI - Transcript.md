0:00
Robert Lange Introduces Sakana AI and Shinka Evolve
So I think a lot of sort of analogies from evolution transfer to scientific research, right, In the sense that we traverse a tree of different ideas or different experiments, and then in the paper we report one path through that tree.
0:12
Speaker 2
When we run LLMS autonomously, yeah, they they tend to just kind of like nothing interesting happens.
0:20
Speaker 1
But oftentimes innovation for a specific problem might require first inventing a different problem, right?
Sort of automatically coming up with this reduction or like this, let's say recursive nature of problem solving is something these systems right now not necessarily have built in intrinsically, right?
0:37
Oftentimes it's easier to generate a lot of solutions than to actually like hard verify them right.
0:43
Speaker 2
The reason why I'm not that worried yet about labor market disruption is I still believe deeply that humans are the source of deep understanding and creativity in the world.
If I didn't believe that, I would be very worried.
0:56
Speaker 1
So I think it's going to be an amplifier of sort of these these latent dimensions humans are great at, right?
1:02
Speaker 2
And I think one of the Rubicon moments is when the the new Transformers architecture or something massive is discovered by AI and we're all using it.
NVIDIA GTC starts Monday in San Jose and it's free to attend virtually online.
There's already been a leak this week of something called Nemo Claw, which is an open source agent platform.
1:23
And if it's real, it could be one of the bigger announcements this year.
So it's definitely worth watching Jensen's keynote For that alone, I'm giving away ADGX Spark.
NVIDIA just hikes the price $700.00.
You probably heard about these memory shortages, right?
1:38
So yeah, it's now $4700, which is very, very expensive.
And Merv from Hugging Face, by the way, she got one for her birthday and she said she literally cried.
So it's a really cool bit of kit.
If you register through my link in the description and you attend at least one session, then you are in the draw.
1:56
This is a massive conference.
Physical AI and robotics are going to be the breakout theme, and Jensen does the keynote Monday at 11 AM Pacific.
The link is in the description.
Don't miss it.
Robert Lange, it's amazing to have you on MST.
2:10
Speaker 1
Thank you, Tim, it's a pleasure to be back.
2:11
Speaker 2
So you working for Sakana?
Tell us about that.
2:14
Speaker 1
Sakana AI is a Japanese AI startup working mostly on Yeah, AI for Japan and at the same time sort of exploring, exploring, let's say novel or ambitious ideas on the research side.
2:26
Speaker 2
It's been around for over a year now.
You're on.
You're one of the founding researchers, right?
2:30
Speaker 1
Exactly.
So, so kind of has been around for now like almost two years, like one in 3/4 I would say.
And yeah, it's pretty fascinating to to look back and to look at the early days and how much the company sort of organizationally has changed.
But in spirit, like we're, we're trying to sort of embrace Ken Stanley's open endedness idea and sort of explore many different ideas which might not get the resources right now in the ML community more.
2:56
Speaker 2
General and we've we've got a few interviews coming out with Sakana that that we filmed here in Japan, so I won't spoil the surprise, but the the CEO is David Ha and David, you know, like there are these epic, you know, giants out there like, you know, Cloon and Stanley.
David Ha is one of these people.
3:11
Speaker 1
David's work has had a lot of influence on my personal PhD, right.
He he a lot of fascinating work on hyper networks and sort of modulation in, in neural networks, but also on evolutionary computation and evolutionary optimization.
And yeah, that sort of also painted, yeah, my path during the PhD.
3:29
Speaker 2
You've, you've released a paper called Shinka evolve and we were just saying that that kind of means evolve, evolve because in, in Japanese Shinka is evolved.
But that's quite common.
That's common thing to do to have these like multilingual, you know, double, double namings in, in Japanese just before we get there.
So we interviewed the Alpha Evolved team and I also interviewed Jeremy Berman a few weeks ago.
3:49
And your paper is, is very much like a more sophisticated version of those in the sense that it's using language models to generate programs and it's doing an evolutionary approach where we generate the program, we refine the generated program and we have an, an evaluator and we do this over several steps.
4:07
And, and your, your approach does many things that that the other ones don't do.
Tell me about the paper.
4:12
Speaker 1
First off, of course, this was partially inspired by Alpha Evolve.
4:16
Beyond Fixed Problems: Co-Evolving Solutions with Shinka
I think it's great work.
I know Alex and Mattei and I think they're doing incredible science.
One thing that sort of is important about sort of using all of these evolutionary LLM driven methods is sample efficiency, right?
So many of these systems sample like let's say 1000 programs for a given task.
4:34
And what we tried to do with Chinka Evolve was try to essentially cut down costs as well as sort of computation evaluation time by introducing a set of sort of technical innovations to this evolutionary search.
And we showed that it's possible with very few program evaluations to basically improve upon like for example, the circle packing canonical result that they showed in their paper.
4:57
And yeah, more generally speaking, I think we're right now at a point or like at an inflection point where these sort of, let's say evolutionary driven LLM systems can really revolutionize scientific discovery.
And yeah, we hope to have made a step forward to making this more democratically accessible, right.
5:16
So the code is open source, available, and yeah, by it's a sample efficient nature, we hope that many people can interact with the system and can make their own scientific discoveries as well.
5:28
Speaker 2
Yeah, that's actually a really important point because I suppose we can use these foundation models.
And first of all, isn't it just fascinating to reflect that we have these amazing models out there that we can access, so like GPT 5 and Glock 4 and they are so much better when you get them to refine their solution in in several steps.
5:48
Why is that?
I mean, I suppose a naive question would be why?
Why aren't they just good out-of-the-box?
5:53
Speaker 1
Potentially, like with enough random samples, right?
It's sort of this monkey typing on the keyboard, they would potentially be able to get there, right?
But in principle, it's sort of coming back to the principles of evolution, right?
In the sense that you need to collect a bunch of stepping stones first and then build on top of them to, to really find innovations or to tune innovations down the line.
6:16
And I think language models with the right sort of evolutionary harness are extremely powerful in terms of scaling up to, to, to make discoveries.
And yeah, I think Jeremy, as well as the Alpha Evolved paper as well as sort of work we've done on like the DAV and Goodall machine, for example, shows that this sort of stepping stone accumulation plus iterative verification and collecting sort of information and evidence from the real world, real synthetic evaluator is really important for that.
6:48
Speaker 2
Very cool and stepping stone collection.
So this is came from Kenneth Stanley is a wonderful paper why greatness can be planned.
And he said that it's it's better to have systems that don't converge.
So in natural evolution, we are just trying all of these different things and greatness quite often follows A diverse path, which means you have to do things which initially seemed quite stupid and then later on they turn out to be incredibly useful.
7:16
We're trying to design algorithms that can kind of allow for a population of slightly weird things and and then we kind of lock in and and converge a little bit.
So we we're still converging though.
So we're still building systems that don't diverge forever.
7:32
What are we losing?
7:33
Speaker 1
One, one thing I find extremely important after having done Shinka evolve is sort of this problem, problem, right?
So with all of these systems so far, maybe except for the AI scientist, which we can also talk about, the problem is given, right?
So you have an evaluator, you have a correctness checker, and you sample programs only on that single problem, right?
7:54
But often times innovation for a specific problem might require first inventing a different problem, right?
So for example, I think in the matrix multiplication result that the alpha evolved people show, you can recursively apply sort of the algorithm to larger matrices.
8:09
So it's actually an important result, right?
But sort of automatically coming up with this reduction or like this, let's say, recursive nature of problem solving is something these systems right now not necessarily have built in intrinsically, right?
So I think going forward, it's going to be really important to not only sort of do open-ended, let's say, optimization of solutions, but sort of do the Co evolution of problem and solution together in order to collect even more diverse stepping stones.
8:39
And to really kick off this this open-ended process.
Because also to me, like 1 of the the big life goals or achievements I would want to see is really having a process that can run not only for let's say a week or many weeks, but like for for years even potentially, right, collecting even more diverse interesting stepping stones.
8:58
Inside Shinka Evolve: Architecture, Islands, and Circle Packing
Yeah, I spoke to Joel Lemon and he was talking about the knighty and uncertainty, which is that machine learning algorithms aren't very good with unknown unknowns.
And, and in a sense, the unknown unknown is talking about these, these stepping stones that might be useful later.
9:13
And when we run these algorithms at the moment, it's the same with LLMS and reasoning systems is that they're very, very good when we give them a specific thing.
And what you're pointing to is we might need to invent a new unrelated problems and find the solutions, which might then be related to what we're trying to do.
9:32
So that feels like a bit of a catch 22 situation, right?
So we're saying, you know, circle packing, here's my evaluation function and I want you to sort of diversify and then, you know, kind of and then converge towards the solution.
It's just, I had the same thought with Jeannie, by the way, that it, it gives you exactly what, what you ask for.
9:51
So you put, put a prompt in and, you know, like a Swiss lake with, you know, with boats on the water and mountains on the side.
And I was thinking, where are the birds?
Oh, I forgot to put birds in the prompt.
Right.
So how can we meaningfully build systems that actually kind of bring in other unknown things that might be useful?
10:07
Speaker 1
I think 1 inspiration or I think I would personally want to sort of research are systems like outlined in in power play or poet by by Schmidt Ober and by by Jeff Cloon and others, right?
So where there is essentially like a set of tasks and a solution generator and both of them sort of Co evolve in this almost like art curriculum play like style, right.
10:29
And I think sort of the in poet, the natural first application was sort of reinforcement learning.
But I think this can now be broadened up to to yeah, science more generally, right?
At least when there is a emulator available to for for running these evaluations and by doing such a cool evolution, you always try to to Max out the the capabilities of that generator while sort of increasing this this convex hull or potentially even yeah more diverse problems while doing so I.
11:04
Speaker 2
Know that there's always the leading thought that even with poet, which was this thing where you had like a population, you had like a load of environments and agents and the environments were in complexified.
So the agents would have a kind of effective curriculum to to learn things and increasing complexity.
11:22
But even then, isn't there a kind of design bias in the system where there's some code somewhere which complexifies the environment step by step?
And wouldn't that also just be designed by the humans?
So it would also just give you exactly what you asked for.
11:35
Speaker 1
Ultimately this comes down to like, the hypothesis that language models can potentially do extrapolation or interpolation, right?
In the sense that even though these things might be in the end designed by humans, there are many unknown unknowns, right, That we humans didn't think of while designing them, right?
11:53
So potentially it is possible for an LLM to, yeah, find a novel discovery simply by us not having thought about it before, right?
12:02
Speaker 2
When we run LLMS autonomously, yeah, they they tend to just kind of like nothing interesting happens.
So depending on the prompt you give them, they'll kind of go a few steps in that direction and then no new interesting novelty emerges.
12:19
And I think even if you wire them eventually with environmental feedback, they they still seem quite parasitic on their starting conditions.
With an LLM, could we build a system which actually adapted to novelty that could actually discover new things?
12:34
Speaker 1
I think it really kind of also depends on what do you give the LLM as a starting point, right?
So for example, in Chinka evolve, we from time on time saw that if you give an initial solution program, which is already pretty optimized on the problem at hand, you still kind of get stuck in, in local Optima, right, where not a lot of novelty is introduced, right?
12:56
While if you start off from like an impoverished solution, there's much more room for diversity.
And I think this is sort of coming back to you sort of what I did before in, in my research name, the meta learning sort of this classical trade off where you can either start out from something very, let's say, unconstrained from like a very simple solution and give much more room for the optimization.
13:19
But this might actually require open endedness in a long time to find a good solution.
Or you start out from something that is already very constrained by inductive biases, let's say.
And then you might be much more efficient in terms of convergence, let's say, but you don't have the sort of open-ended, big novelty sort of benefit from it.
13:43
Speaker 2
Yes, I suppose where we want to get to is building systems which are not designed by humans.
So for example, if if I'm leveraging my deep understanding, you know, LLMS are really good if you, if you understand something deeply.
And similarly, we could kick off a Chinka revolve and we could, we could put a starting solution in there which leverages my understanding.
14:04
We want to have AI systems that, that anyone could use.
So just a non expert could say I want to solve this problem and and it will solve the problem.
We should talk about the the evolutionary approach, right?
So to maintain diversity, you had a population of programs and they were separated into islands.
14:22
From Semantics to Algorithms: LLMs as Algorithm Designers
Tell me about that.
14:23
Speaker 1
The way how Shinka evolve, similar to alpha evolve works is you keep an archive like a database of programs and then you sample parent programs with a set of sort of inspiration programs.
And then you ask an LLM to basically make an improvement to that program, right?
14:39
So to provide code to edits or rewrite an entire program or to potentially even crossover different programs.
And then you, you basically you query the LLM, you get a program out and you evaluate it on the problem it had, right?
So for example, increasing the sum of the radii of a bunch of circle in a square, you run this basically each time collecting evidence from the evaluator, adding it to the database, and then sort of repeating this process.
15:06
And you don't do this sort of sequentially, but you do this in parallel for many different programs.
And each time sort of a program is added, you essentially try to diffuse the knowledge that was collected by the program across the entire sort of database, right?
So one way to think about this is you have a tree, a tree where each node in the tree represents a program.
15:27
And then you you sort of branch off of it based on the parent notes, right?
And interestingly, like these approaches do tend to scale, but ideally, we can make the scaling hub at a faster rate, right?
15:42
And this is something we tried and Chinka evolve by sort of doing a bunch of innovations including sort of model ensembling.
So we're not using just Gemini, but we're using basically all frontier model providers and figuring out a smart way how to use each model for a given current, right.
16:03
So if you have a certain program and some situations, it might be better to use a sort of GPT model.
In other settings, it might be better to use Gemini model.
And we sort of introduce a sort of adaptive prioritization scheme that can adapt sort of the evolutionary algorithm on the fly while running the the algorithm.
16:22
And this sort of also comes back to the naming, right?
So Shinka evolve, evolve, evolve kind of means that this evolutionary algorithm that we apply using LLMS sort of also Co evolves at the same time while we optimize the programs.
16:36
Speaker 2
And on this, while we're on this circle packing problem, so you, you had this plot showing how it converged and it seemed to converge quite quickly.
So and we'll show the plot on the screen now.
So very quickly the performance jumped up and then it slowly converged.
And you said in the paper that it was using three, I think, 3 core innovations.
16:54
And my thinking was, if you ran this 50 times, would it be the same every single time?
And how to what extent is it thinking outside the box?
You know, Sebastian Bubeck is always posting on Twitter talking about how GBT 5 has just, you know, discovered new things.
17:11
And there's always the question of, well, is it just searching the Internet?
Is it just finding things that have been found before?
And, yeah, combining things together and in a new way.
But could it really think outside the box?
17:22
Speaker 1
Yeah, I think this is umm, almost like a subjective question, right?
So First off, I don't know all problems on the Internet that try doing circle packing, right?
But what I can see in the tree that we also depict is, umm, there's for example, like a crossover operation between two programs happening where sort of, umm, uh, different concepts are combined, right?
17:42
So one important part is, for example, the, the initialization of the circles.
Another one is like the optimization.
So basically like a constrained optimization program is executed.
And then the final part is basically like a reheating stage, right?
Where noise is added and sort of more try to be squeezed out.
17:59
And to me, like this sort of propagation of information through the tree is one that's really, really fascinating, right?
Where in some sense these stepping stones are actually used and so in a complementary fashion, right?
And with regards to rerunning the program multiple times, right, Of course there's some stochasticity on that, right.
18:19
So we're using language models and sort of due to like the, the queuing device scheduling on on their server side, basically we can't get rid of all the all the noise we we've seen that at least for the general quality of the solution.
So what is arrived afterwards, it is possible to re obtain this, but sometimes with a different program like or most of the times just by stochasticity, right?
18:42
So it's not like there's for many problems, there's like not one solution that achieves that score, but there is like a spectrum or like a, a region, let's say in the program space that that for example, it's the same, right?
I think one thing that was very interesting about the circle packing problem sort of also coming back to the problem problem that I discussed initially was that originally we, we used a formulation where the correctness is checked with like a very tiny amount of slack, right?
19:11
UCB Bandits and LLM Mutations for Adaptive Code Evolution
So the, the circles could overlap a tiny little bit and then afterwards we, we, we sort of reduced the rate AI and the solution was exact, right.
This didn't change the score by too much.
So it's still state-of-the-art, but it was essentially like a proxy problem.
19:27
When he then reran the, the Shinka evolved on the exact setting and we found that it took a little bit longer to actually obtain the same quality of a solution.
So I think this already points a little bit in this direction of what I discussed in the beginning, like sometimes sort of surrogate problems might actually be extremely valuable in in making such discoveries.
19:48
And having an automated way for designing these surrogate problems in an efficient way might be something really important going forward.
19:55
Speaker 2
Yeah, that's absolutely fascinating.
It reminds me of support vector machines where we'd make the optimization tractable by introducing slack variables, and you can think of that as a kind of surrogate problem.
But then I'm thinking, well, would Shinker evolve?
What Alpha evolved would it know to introduce a surrogate problem?
Because, you know, as designers who understand, you know, we can think outside the box and we can do stuff like that.
20:16
Because presumably if the fitness function had the constraints that there were no circle intersections, then it wouldn't.
It wouldn't occur to the algorithm to come up with a surrogate problem.
20:26
Speaker 1
Exactly.
Yeah, this is a big limitation right now, right?
So at this current point in time, we take the problem to be fixed and we optimize for that problem.
But when you think about humans, we're really, really good at sort of inventing our own problems, right?
Or reformulating the problem so that we can actually sort of work with it, right?
20:45
So I think a lot of sort of the innovations in, let's say, mathematics come from taking a very different perspective on a problem, right?
So taking sort of number theory and applying it to linear algebra or the other way around.
And I think right now these systems are not yet at the point of, of of achieving such level of let's say transfer.
21:06
Speaker 2
Yes, and it reminded me, I spoke to Lyon about this.
You've got this Sudoku bench and a lot of folks watch cracking the cryptic YouTube channel and that's exactly what they do.
They invent new problems based on abstractions that capture the essence or aspects of the problem you're solving.
21:22
And then they do something which is similar to Shinka revolves.
They do this kind of evolution where they take these different solutions and and they kind of combine the the best aspects of both of them and they forge a divergent path to a new solution.
And that seems to be the essence of of what we need to do.
21:37
Speaker 1
Yeah, for sure.
I, I mean, there is some work also by Jeff Clune, Shengrenhu and Sunglu on automatic automated capability discovery.
So there they look at language models that generate tasks, right.
But it's in a, let's say, unstructured way in the sense that it's not done in order to enable the solution to 1 target problem, right.
21:58
And I think sort of doing these connections is going to be very fruitful down the line.
22:03
Speaker 2
Very cool.
Now the other thing, we'll show the graph on the screen, the evolutionary graph.
So for the circle packing problem, I was looking at that and first of all, it looked incredibly posimonious, which is good.
It looked like it had found an optimal path to the solution very quickly.
And I was thinking in my mind, well, maybe there's some natural pattern.
22:20
There's there's, there's, there's something about that that we could use in the abstract to guide the evolution in the future.
But the other thing I'm thinking about is right now the problem with machine learning is that we don't really have semantics baked in.
So what we're doing is we have a verifier, we're looking at the rewards, and we're sort of like doing patterned exploration and we're taking steps towards the, you know, towards the target.
22:43
And I love mechanistic forms of reasoning where we actually know something about what the programme components mean.
And the reason this is important is when we're merging together the best performing programmes from two different islands, that's a kind of first order interaction and it might not make sense to merge them together.
23:01
It's wonderful that LLMS, you can give them any pairs of programs and it will find a way to merge them together.
But wouldn't a more principled way be of there's, there's some kind of semantic primitives here and we know they fit together.
So there's this Lego analogy that we're kind of building up based on principles rather than forging a path based on the performance.
23:22
Speaker 1
Yeah, that's a good point.
So one thing we do in Shinkai as well as we keep essentially a scratch pad.
So each program is being summarized and then from the program summaries, we keep sort of a set of global insights, let's say.
Then we're shared or like extracted from these programs.
23:39
And then based off of the scratch pad, we construct sort of meta recommendations that then become part of the system prompt, right?
So that way you can try to sort of semantically grasp some of the discoveries.
But a general problem, which is again sort of task dependent is thereby you sort of diffuse that knowledge across the tree, right?
24:00
But sometimes you want things to be much more isolated, right?
It's always like a trade off where you somehow have to find four year problem the right position on the spectrum of how much knowledge diffusion do you want to have and how much sort of, let's say hard islands of programs do you want to have, right?
24:18
The Future of 'Vibe Research' and Democratizing AI Discovery
And yeah, we're trying to make steps in the direction of sort of automatically adjusting this in an optimal way.
But again, it's very program sensitive.
And then sort of I think another point where you're already sort of going into is sort of Jeremy Jeremy's solution to arc AGI, right?
24:35
And sort of doing solution evolution in the instruction space, right, instead of the program space.
I do think that this is something important and we're like I said, with like the construction of this meta scratch pad trying to do sort of both at the same time.
24:51
Again, it's a problem dependent.
Like I played around a little bit with arc AG1AGI1 and arc AGI 2.
And I think on arc AGI one actually the the transform sort of program direction is actually quite effective, right?
It's like Jeremy said, it's deterministic and it's easier to sort of get clear signal to improve on during your evolution process.
25:11
While on others like arc AGI 2, like this whole sort of semantic evolution seems to be more efficient.
So I think ideally we we can get a system that can automatically some sense decide whether or not it wants to take like a programmatic approach in settings where it's actually feasible and easier to to bootstrap off or it takes the semantic approach of evolving instructions or like LLM driven input output mappings.
25:36
Speaker 2
Yeah, it's, it's so interesting because you know, like a, a symbolic AI person would say, oh, I don't like connectionism because it doesn't, you know, the only semantics in connectionism is this notion of similarity.
It doesn't really understand things.
So, so they would say, well, just just start with an entity relationship graph and then just kind of build up using, you know, composition and 1st principles that that that doesn't work, right?
25:59
So we're using neural networks because they're incredibly flexible and they understand a lot of things about the world, but they don't have the kind of constraints that we want.
So what we do is we use these tricks.
So Jeremy evolved programme descriptions on your programme selection.
You had a semantic novelty detection, you know, using like a.
26:17
Speaker 1
Embedding based similarity, yeah.
26:18
Speaker 2
Yeah, so you had like a kind of self similarity matrix and you know, based on the the cosines and and and indeed you've got this meta scratch pad.
So what we're seeing is this fascinating spectrum of possibilities where still using neural networks, you can imbue semantics in using all of these different tricks, but they all come with trade-offs.
26:36
Speaker 1
Yeah, for sure.
Like I think it's, it's kind of interesting.
We we've had a long period of computer science where algorithms were sort of designed by humans, right.
Then we had sort of this Andre Kapathy and software 2.10 paradigm where like we trained neural networks that then performed a certain function.
26:53
And now we're sort of at this point where we're using LLMS to design algorithms or solutions more generally, right?
27:00
Massively Scaling Shinka: Meta-Evolution and Open-Ended Adaptivity
And I think actually like, even though like large frontier language models are extreme, like let's say black boxes or it's very hard to get a full mechanistic understanding of them and the outputs can be right, the programs, the instructions and so on, right?
27:16
So I think it's opens up a very sort of new paradigm of doing research or basically doing anything right, if you, if you think about it.
But I think we're we're just sort of at the starting point of figuring out the the right user interface for that.
27:30
Speaker 2
So the other innovation in the paper was using UCB, which which is upper confidence bound.
It comes from the multi armed bandit literature, which is this problem where you can pull these these levers.
And at the beginning you don't know which levers to pull and, and over time you kind of reduce your uncertainty and you can kind of pull the ones that work.
27:49
But there's this exploration exploitation dilemma and you've implemented that for figuring out which LLM.
So it could be Gemini, it could be like, you know, Grok 4 or something to figure out which one to use.
28:02
Speaker 1
We're using like a model ensemble, right to propose program mutations and umm, intuitively one could say like the the best frontier model on on sui bench is always the best mutation proposal model.
But that's actually, in practice, not always the case, right?
28:20
And in general, it's extremely hard in this evolutionary setting to assign clear credit to a single model, right?
So you have, for example, like one improvement is implemented by GPT 5 and then the next one is implemented by Sonnet 4.5.
28:36
And it's unclear basically if the performance gain you get from the second mutation actually originated from GPT 5 sort of collecting the first stepping stone or from Sonnet 4.5 S instead of sort of uniformly sampling models.
What we do is we implement this bandit based approach where each model is basically one arm of a bandit.
28:57
And then we look at how often did this model improve performance of a sort of parent node by creating a mutation.
And we then adjust sort of this posterior probability to sort of first explore all arms once, right?
29:12
And then essentially change over the course of time in order to prefer models that sort of yielded improvements before for similar notes.
29:21
Speaker 2
The great thing about using a UCB like algorithm is you can.
It actually has a theoretical regret, which means it's not.
It's like only log worse than the optimal switching path, if that makes sense.
But if I understand correctly, UCB is based on a sort of like a global rating, like a mean score of every single LLM.
29:45
And I think what we want is to have more of a contextual switching and decision, which means we know for this particular program Gemini is better.
And do I understand correctly at the moment that it might converge to a single frontier model and then in a nuanced situation we might still get the wrong model?
30:02
Speaker 1
So in general, like there is some amount of probability associated like allocated to all models, right?
So it's not like it can just peak on one model and then you stop using the others, right?
So there's still a chance for open endedness and serendipity, if you will.
And we in general like for the problems we consider, we, we haven't seen that like 1 model clearly dominates all the others, right?
30:24
We've seen then it really depends on the course of this evolutionary process like which model is better and UCB or like the the bandit approach that we take dynamically adjust this in in an efficient.
30:37
Speaker 2
Way.
And would it be possible in the future to use an LLM to make this judgement?
30:41
Speaker 1
Potentially in some sense, in that case, again, you think of the LLM as a surrogate model, right?
In some sense you can think of like a Gaussian process as a surrogate regression model.
And there has been some work sort of showing that language models can act as surrogate models.
30:57
And the real question to me is like, how do you represent the information to the LLM, right?
In the sense that if you use like the raw programs and their fitness evaluations, you, you quickly run out of context, right?
So you need some amount of compression in order to present the information the right way to the LLM in order to do this prioritization of the models.
31:16
Speaker 2
I hadn't appreciated how long the context is when I was thinking, you know, could we use like an 8 billion LAMA model and we're doing active fine tuning.
So we're saying, I just ran it on, you know, I just ran this program on Grok and and it got this score.
31:32
Yeah.
And and then over time that you know this thing for the given run of this evolution, it will kind of know that Grok is good at these problems.
31:40
Speaker 1
Yeah, potentially.
I'm not sure like how efficient this fine tuning is if if we're only evaluating like 150 programs.
But in principle, one could imagine, I think it's on the engineering side, not necessarily like the prettiest to do.
Yeah, it could.
31:55
It could in fact happen.
But I think like for all of these things, we started out sort of with the, let's say, most intuitive algorithmic component that we had.
And UCB was one that really did the job here.
And yeah, much credit to Eduardo Sateen, who introduced us to to Shinka.
32:12
Speaker 2
So let's talk about the the diffs and the mutations.
So we we generate programs and I think you folks were inspired a bit by alpha evolve.
So they actually had this gating where where you kind of gate part of the code, which is mutable.
32:29
Tell me about all of that.
32:29
Speaker 1
The program is just, let's say, a long string, right?
And in order to to make sure that certain parts which are sort of essential to the evaluation, for example into the imports and so on, we're not sort of deleted by the LLM mutations.
There are so-called markers which basically state which parts of the code are mutable and evolvable.
32:50
And it's easy to like programmatically sort of make them actually immutable when you get a diff proposal and these will not be changed.
So only the the rest of the the code snippet will be changed.
We sort of implement a type of rejection sampling with reflection approach where if an LLM by chance, for example, tries to mutate this part, it's going to be rejected and you resample a new proposal.
33:14
And yeah, thereby you, you can somewhat mitigate certain security or safety problems.
And yeah, get a robust sort of mutation.
One of the sort of I think the the bigger questions is how can you turn this from a single file mutation setup to a multi file mutation setup.
33:33
So working on entire code bases, in principle you can represent many code bases in a single file, right?
But the hierarchical structure might be actually useful.
And there are some ideas from let's say ader this, this coding tool where you construct like a repository map and sort of have some level of abstraction, but they also come again with positive and negative trade-offs basically.
33:58
Speaker 2
I love Ader by the way.
It feels that in the future the, the, you know, like the code generation systems will, will actually resemble Schinke revolt.
And if if you think about it, it'll be using some kind of git repo.
34:13
Shinka Evolve: From Agent Scaffolds to the ARC Challenge
Maybe cursor already does this because in cursor you can restore previous checkpoints, but it can be exploring different branches and and merging checkpoints together.
And, and you know, obviously you just say in natural language what you want to do, but we didn't talk about mutation, by the way, so we just spoke about diffs.
34:30
And there's also an option to do the full file rewrite exactly.
But there's also this notion of crossover.
So how does that work?
34:36
Speaker 1
A small innovation on top of Alpha Evolve and where I believe they only use sort of stiff based mutations is that here we wanted to have more flexibility to entirely rewrite the program, right?
To come up with a completely different stepping stone, if you will.
So again, there you can make parts of the code mutable, but instead of proposing, let's say a patch to change certain parts of it, we essentially rewrite the entire program.
35:02
And this sometimes is helpful, right?
It's not always like a clear benefit, but it, it allows you to essentially get more diversity into the search, right.
So this is 1 type of mutation next to sort of the stiff patch based approach.
35:19
And the other one is a crossover mutation where we sample basically and not only a single parent program, but sort of two different ones.
And we ask the, the system to sort of make a complementary improvement.
And again, on some problems, this is really helpful and on others it's not.
35:34
But in generally we found that sort of having a diversity in terms of operators is also helpful in discovering new things.
And I wanted to to sort of follow up on the point you made before about this sort of being a new paradigm.
I think so too.
I'm really convinced.
I think right now we're sort of at the beginning where we, we still think a lot about sort of this chat assistant interface as the way how we interact with LLMS.
35:58
But it's most of the times inherently single threaded, right?
So we're sitting in front of the computer, we're interacting with the chat, we're seeing sort of changes as they occur in the editor, we accept them and so on.
But I think this is sort of also just a stepping stone towards sort of a more, let's say, distributed way about thinking about research optimization and so on.
36:19
So I like to sort of think of vibe coding, vibe chatting.
And on the other hand, we have sort of vibe optimization and vibe researching where sort of my ideal future scenario is 1 in which you as a researcher sort of during the day Co work with like a system like Shinka or the AI scientist.
36:38
You sort of steer the ship like a shepherd in some sense.
And then during the night, you you, you, you press play and you go to bed.
And in this in the background, you've multiple experiments running and automatically new ones being proposed by LLMS, evidence being accumulated.
36:54
And then in the morning you come back and sort of you have an multi threaded sort of system running in parallel.
And you're more like the shepherd of the ship than the the person actually executing experiments and analyzing.
Oh yeah, you're still analyzing, but you're not executing.
37:09
This is happening sort of by the system itself.
37:11
Speaker 2
Yes, and increasingly this might be semi supervised or even proactive.
I mean, you know, there's that new product from Open AI where it knows what you're interested in and while you sleep it's going off and you know, fight your pulse.
That's right.
And you know, we're in the situation now where we're reasonably technical people.
37:27
So, you know, MATLAB and Mathematica, they're, they're supremely powerful, but you need to know how to express problems precisely.
Whereas I can imagine a future where we express problems just in natural language or maybe just based on our interactions with language models.
37:46
The platform knows what we're interested in and it can just go and find things on our behalf because this is about democratizing this technology to people or who perhaps don't know exactly what they're looking for.
37:56
Speaker 1
I think one of the bigger problems there is sort of this verification aspect to it, right?
In the sense that often times it's easier to generate a lot of solutions than to actually like hard verify them, right?
Language models are capable of doing sort of soft verification, looking at code and sort of latently running like a like a stack trace of execution, right?
38:16
But it's not exact, right.
And I think sort of these notions of reward hacking and sort of not doing real discoveries, but sort of shortcutting them is 1 where we need to put more time and effort into to figure out how to make sure that this actually moves in the right direction, right?
38:34
And I would hope that language models at some point can do this efficiently themselves, right?
So either implementing in code or latently doing it.
But this is also like part of the problem problem, right?
It's not only coming up with the problem, but also with the automatic verification at the same point.
38:51
Speaker 2
Yeah.
Isn't it a tantalizing idea that there are natural patterns in the world and the building blocks to construct novel solutions are already there, right?
And, and maybe they're there for a reason.
Maybe they just reflect natural regularities in in the universe because there's always this question of, you know, intelligence is about adapting to novelty.
39:13
So the world is always changing and the world tomorrow we'll have things that we can't explain, you know, with our with our knowledge today.
But we do have like abstract knowledge that could be easily recombined to explain the future.
And LLMS might already have those building blocks.
39:30
Speaker 1
Yeah, for sure.
I think like in some sense, the more you think about sort of Occam's razor applying to everything in our world, like let it be language or let it be sort of science is, is pretty interesting because like these artifacts now go into our language models of today and potentially there is some amount of this being captured.
39:49
I think though it might also be an unactive bias that leads to a local optimum at some point, right?
And you need more complexity.
But I do think like with systems that sort of do this evolutionary mutation sort of style approach, you might still sort of push the system out of these local optimum eventually.
40:04
Speaker 2
Yes, and then there's also the notion of the importance of adaptivities.
So this is what surely says intelligence is and since we've had these models that actually do adaptivity at inference time, so things like test time, active fine tuning and the reasoning models and and so on, they started getting non trivial performance on arc.
40:23
Now it's very, very expensive to have adapting huge foundation models.
You know, it's it's just a practical concern why we haven't done that yet.
But what we can do is build systems like shrink or evolve that leverage the best of both worlds.
So they leverage frozen foundation models, but they give you adaptivity.
40:42
And the purpose of adaptivity is to respond to novelties, to create new building blocks, synthesize new building blocks in this principled tree like structure that allow us to adapt to novelty.
So we are having our cake and eating it, I have to say.
40:56
Speaker 1
I found it very interesting that Jeremy basically in your podcast when you asked him about Chinka, was saying like he doesn't believe that there are a lot of sort of percentage points to be gained by using a system like Chinka, but you can make it much more efficient, right?
That was sort of the gist of his answer.
And to me it's like once you have made it much more efficient, you can scale it up again, right?
41:15
So if you essentially have a cheaper system that can generate many more sort of instructions, I would expect that by the nature of open endedness, you might get some amount of improvement out of it.
Right now I don't have any evidence for it and I would love to collect that evidence.
41:30
It's again like the magic of open endedness that comes into play that as long as sort of these training examples of our Gage, I give you a good signal for a final Test submission, you should be able to to progress.
41:41
Speaker 2
Yes, and that and that is a great segue because certainly on the circle packing problem it was so sample efficient that in less than 200 you know interactions with an with an LLM you converged on the solution.
But I was thinking that great.
But it's still quite dependent on the starting conditions.
42:00
You know we talk about this design bias and and and so on.
So what we put in is very important.
But now what we could do is scale out so we could run this 1000 times and we could have another process which prompts, generates, breeds the starting conditions because, because every time we run Chinka Revolve, what it's doing is it's it's searching parts of the epistemic tree.
42:21
And what would happen if we just scaled that out massively?
42:23
Speaker 1
We haven't tried, but you could even start with like an empty program, right, which would be, it would be basically the same, right?
And then you would branch off of that empty program I would expect.
Yeah.
We haven't done this simply out of sort of cost and time reasons.
But I do think in many ways sort of this is the question that will towards like this true open-ended vision of running a system for like a month or so, right.
42:47
Really trying to squeeze this out.
Yeah, I'm not sure if we're entirely there yet, but I will do my best that we will.
42:52
Speaker 2
And the reason this is interesting is we know as a practical matter that we can't start with nothing.
If we were just sort of like starting from the most primitive building blocks, the search base would just be huge and there'd be no learning signal.
So we know we need to start a little way up the stack, but we can massively parallelize that.
43:09
So let's say we have 1000 different instantiations of Chinka revolt.
It doesn't have to be embarrassingly parallel.
We could still have some sharing.
So during their execution, we can still have a little bit of like crossover and, and and and maybe then we could, we could run or the Chinka evolve instantiations in a, in a similar kind of meta evolution loop.
43:29
And my suspicion is, Contra Jeremy, I agree with you.
We know there are diverse stepping stones out there that could dramatically, dramatically improve many of these solutions.
We simply haven't scaled it up yet.
43:41
Speaker 1
Yeah.
I also believe that using a system like Shinka Evolve could be able to sort of automatically detect whether or not like an instruction based optimization approach for a given problem or a transform based approach is actually the right thing to do.
43:57
And sometimes potentially it's like even the mixture, right?
There's some things you can probably easier even articulate in Python then you can articulate in in sort of language, right.
So I would be really interested in sort of exploring that.
44:10
Speaker 2
Yeah, I mean, you said earlier about Jeff's clear what, what was Jeff Kleen's paper the the thing that generated?
44:14
Speaker 1
It's capability discovery.
44:16
Speaker 2
I did speak to him about this at Aneurys, but something like that could be fascinating as well.
You know, where we're also generating the problems and solutions and then kind of moving the back end.
But I, I think the way this will land commercially is there'll be a new type of GPT where everyone is solving different types of problems and, and the system, it'll be like a kind of chinker revolved with a massively distributed version where mathematicians are using the platform over here to solve this problem.
44:41
And it will see commonalities and it will kind of like link them together because you need to leverage like human creativity in this process as well, I think.
44:49
Speaker 1
Like a big challenge going forward is going to be like how do we change our incentive system for this to actually scale, right?
I think like, for example, some amount of economy will be needed or some amount of mechanism design in order to make sure that everyone is still happy to engage in it, right?
45:06
Navigating AI's Impact: Job Market and Human Adaptability
So maybe we're going to have many more leader boards for whatever is numerically sort of scorable.
And I think this this will be really, really interesting to see how sort of compute these automated agents, human shepherding and steering will ultimately sort of change and revolutionize science and I guess society more generally.
45:26
Speaker 2
And Rob, looking at the future, we've got a load of people in in San Francisco that's that want to scale language models and they are adding in implicit forms of adaptivity and composition so that they're building controllers and they're doing reinforcement learning with verifiable feedback and so on.
45:41
I think that you subscribe to the slightly different idea that that we need to be far more open-ended and we need to be using evolutionary algorithms and so on.
But do you think that they are on a path to nowhere?
Do you think they might change TAC?
I mean, where?
Where is this going?
45:56
Speaker 1
So I actually think that these things can be complementary, right?
In the sense like, let's say you find tuna model to be like a circle packing expert, right?
So I do believe that mixing in sort of different sort of RL fine-tuned models into sort of the ensemble of models and then having a good way to adaptively select which model to use is not a bad idea, right?
46:19
So to me, I just very fully subscribe to this philosophy of open endedness.
And reading Ken's and Joel's book was really like a fundamental moment in my life.
And I want to see how far we can push this.
46:34
And I think we're we're not yet at sort of convergence where either the capabilities of the models has converged or the the way how we scaffold around them or the way how we humans interface with them.
So to me, they're really like these three points like model capability, model scaffolding, and sort of the user interface.
46:57
And I think we have a lot still to push on all three angles.
47:01
Speaker 2
Beautiful.
The only thing we didn't talk about was we spoke about the circle packing problem, but you also applied it to a few other things.
Can you tell us about that?
47:07
Speaker 1
So one thing we did was we sort of used a framework called ADAS, automatic design of a Genetic system, where basically instead of manually writing an agent scaffold, you use an LLM to write agent scaffolds for a specific task, right?
47:23
So what we did is we looked at mathematics tasks, So Amy and we used Chinka to evolve basically an agent, right?
So using an agent to evolve an agent.
And we found that there we could dramatically improve sort of the performance of very cheap models like GPT 4.1 Nano.
47:42
But the agent scaffold was also able to either like generalized to other language models or to different years of of Amy, right?
There was one application, one important other application that we did was to ALE bench.
ALE bench is basically work done by other folks at Sakana including Yuki who's also part of the paper which is considering heuristic sort of programming contact contests sort of previously done and executed by Atcoder, which is like this famous Japanese competitive programming organization.
48:16
And we sort of showed that Shinka can also work very well as a Co scientist.
So basically we we took initial solutions obtained by an ALE agent that was previously designed and then we optimized on top of these initial solutions with Shinka and showed that on one of these sort of programming tasks, if the combination of this agent and Shinka would have competed in the challenge, it would have ranked 2nd place basically.
48:42
The AI Scientist: Amplifying Human Creativity and Understanding
So I think there's some evidence that Shinka can work as a Co scientist and not only for LLM agents, but potentially even for humans like we discussed before.
And then finally, the final application that we looked at was designing sort of mixture of expert load balancing loss functions.
49:02
So at Sakana, we've done some previous work called Discopop.
I think we discussed this during the last podcast we did where we're using LLMS to design objective functions.
And back then we did it for preference optimization and post training.
And here we did it for load balancing of mixtures of experts.
49:17
Also there we found that within, I think like even only 20 sort of generations, we were able to sort of explore, let's say, not only a single objective function, but sort of, let's say, a convex hull where there are different trade-offs between sort of performance and load balancing and so on.
49:35
So I think this is another application of Shinka where it's not only basically about sort of finding the best solution, but essentially illuminating a program space where there are always potential trade-offs between like let's say, for example, runtime and the quality of the circle packing, right?
49:53
And having a system that can explore all of these is important as well.
49:57
Speaker 2
I'm very excited to see you apply this to the Ark Challenge.
Like what are, what are?
What are your thoughts about that?
50:01
Speaker 1
I still need to collect results, so I don't want to make any claims like or hard claims before having done this.
But I would hope that there is some chance of for sure improving sort of the the cost of these systems and then potentially even performance.
50:17
But yeah, to be seen.
50:18
Speaker 2
Oh very excited.
So you've done some experiments.
Exciting news is potentially coming I.
50:22
Speaker 1
Started looking into it.
50:24
Speaker 2
Yeah, and I mean, what, what are your thoughts in general about about ARC though?
50:28
Speaker 1
I think it's great.
I think it's it's really important and I think it fills an important gap.
And I do really deeply respect Francois and sort of read the paper when it first came out and no one thought of actually being able to to get numbers above 10% right.
50:47
And it's also pretty fascinating.
So on a society level, how far we've come since then.
And sometimes while you're sort of deep in the say, battle mode or work mode, you can forget where you were one year ago.
And then just looking back, it's it's pretty amazing also how far we've come since 01.
51:06
Speaker 2
It's insane.
I think Francois doesn't get enough credit because it's such a good benchmark and not necessarily for reasons people think.
Because Francois is always saying that we need to have a benchmark which is easier for humans and hard for AIS.
And, and in a sense, that's not quite the case.
51:21
I, I said when Arc V2 came out that it's actually very difficult for humans that, you know, there is one task where Duggar was stumped for about 15 minutes with there's three of us looking at it and we just, and it's one of those things that depending on your perspective, you might get it straight away or you might not.
So there's that criticism and people have said that Arc V3 is even harder, you know, but I think that's rather missing the point.
51:41
I think he's saying that with with a lot of these competitive coding problems, the data set is contaminated.
These are problems that have been solved before in part or in whole, which means when you look at the epistemic tree, many of the building blocks for solving them are very high up in the tree.
51:59
He's looking at these problems that there is very little data set contamination and and they need to be solved from very abstract building blocks.
So you're starting much lower down the tree and you're synthesizing a model by composing together very abstract building blocks, which is the essence of intelligence.
52:16
Yeah.
And, and I think for that reason, ARC is, is really kind of pushing us to build adaptive systems, which we could say are intelligent.
52:23
Speaker 1
Yeah, I agree.
I mean like in many ways, I'm really looking forward to the next years and seeing how far we can push this.
And then also how much generalization we can get afterwards, because I, I believe like when you look at sort of the more recent models, they're getting much better at the transform style code evolution or outputting for ARC than they are on the instruction based level.
52:46
And I think this might already be like a small sign of some amount of overtraining on ARC ATI one at least, right?
I do believe there are some aspects of work which will be automated before it comes to sort of fully assigned automation and the type of work I'm doing.
53:05
But I could imagine that certain parts of the dimensions that I deal with everyday are for sure going to be hit by AI.
53:13
Exploring AI's Philosophical Frontier: Alien Artifacts and Autopilot
And then the question is, are there going to be new dimensions opened up that we as humans will fill in, right?
And I think what I said before about like shepherding and so on, I really hope that that's the way forward, right?
In the sense that humans are the ones steering the ship while just being massively amplified in their productivity.
53:32
Speaker 2
Right now, I am not really seeing the kind of job market disruption that was being predicted.
I know from personal experience that in in a sense it's made it very difficult to hire people.
You know, script writers use ChatGPT.
53:48
I can spot it instantly.
And, and writers and copy editors are actually in more demand than they were before fixing all of the crap that's been generated with ChatGPT.
And there's the cloud analogy as well.
So, you know, IT system administrators who were earning, you know, £60,000 a year in the UK, they rebranded as, as cloud DevOps engineers and they more than doubled their pay.
54:13
And people are very adaptive.
They, they see new trends, new bandwagons and, and they just adapt and, and they add value on top.
And that has been the trend or, you know, for a very long time.
Do you think that AI is going to be so transformative that it will transcend people's ability to adapt?
54:30
Speaker 1
I think it's just a question of speed, right?
So I was talking about sort of cultural evolution and technological evolution.
And it seems like we humans, we need more adaptation and more time to, to get used to the technology to carve out these niches where we we can fill in.
54:48
And it's complementary, right?
So First off, I, I think we're, we're still not at the ceiling of the sort of technological progression, right?
So maybe in a couple of years we will need less of sort of slop editing like you said.
But I do think we, we need some more time to adapt to the different modalities of interacting with these systems, right?
55:07
I think everyone can sort of interact with a chat assistant, but I think this is the most sort of naive form of interacting with AI agents, for example, right?
So yeah, I think we need to get the pacing of all of this right, and we need to do much more exploration in human machine interfaces, UIUX design, and how to make sure that humans sort of feel or feel fulfilled during this experience.
55:37
Speaker 2
This is particularly relevant because, you know, you were behind the AI scientist paper and there's now a version 2 of that.
Allow me to be a tiny bit sceptical.
You know, we were talking about when we evolve systems to do a, to do a particular thing.
And at the moment it feels like as good as they are, they are still quite parasitic on the instructions and intentions of the human supervisor.
56:01
So it's very much an exchange between the humans and and the system because the implication is that in the future we might have systems that are so autonomous and so open endedness and can figure out valuable things to research that humans wouldn't be needed anymore.
56:19
And the reason why I'm not that worried yet about labour market disruption is I still believe deeply that humans are the source of deep understanding and creativity in the world.
If I didn't believe that, I would be very worried.
56:33
Speaker 1
I agree.
To me, like the AI scientists, like V1 and now V2 are sort of glimpses into a potential transformation.
But I fully agree.
In order to make really big scientific breakthroughs, like multiple of them, like every day or whatever, you still need humans in the loop to sort of either seed or guide the direction in which to explore or to to verify, check and actually, yeah, transfer these insights, right.
57:02
So I think it's not going to be like all ML PhDs will will be unemployed.
It's it's more going to be a sort of core evolution of humans with this technology and potentially like in an ideal future for me, like it will allow humans to focus on what they're really, really great at, right?
57:21
So I think it's going to be an amplifier of sort of these these latent dimensions humans are great at, right?
I think something that's critical is that we as humans try to interact with these systems as early as possible in order to to actually like have influence and ownership over like this development process, right?
57:38
And it's ultimately collective intelligence that will shape all of these systems together.
57:45
Speaker 2
And do you think these systems can become incredibly sophisticated, such that they are, you know, somewhat detached from humans?
57:54
Speaker 1
Well, I mean with the AI Scientist V2, we sort of released that one paper that we submitted to in ICLEA workshop was able to sort of pass the acceptance threshold before method review.
So I do think at least for sort of workshop level contributions, we're getting there.
58:15
While not every submission in AI Scientist paper does is or is reaching that threshold.
We're we're at the point where we can even talk about sort of noisy review processes and this actually being yeah, something that as long as you have a large budget, you might get something out of it.
58:33
I think going forward for the bigger innovations and so on, for now, you still need humans, but we're sort of at the GPT 1 moment of, of making this sort of a reality.
And potentially in 10 years, this is going to look very, very different once they're sort of also the infrastructure for it has been built up, right?
58:51
So there are places like periodic labs, right, which sort of now are building like real physical labs with robotic systems to automatic automatically sort of execute experiments.
This will take some time, but it is sort of imaginable for sure that as we sort of do RL on these types of systems and we actually also account for negative results and for actual like hypothesis testing.
59:17
So getting these systems to be a real good hypothesis testers with verifiers in the loop that we might be able to unlock many more capabilities.
59:26
Speaker 2
Yeah, I mean, I suppose I, I don't want to sound like a Luddite.
So it's entirely possible that this is just, you know, I, I don't have the imagination to think about the future.
So it is possible that in the future that these systems might understand very deeply and be creative.
59:42
You know, I think right now the problem is they only understand things a few levels down in the epistemic tree.
So they can do some surface level recombination and they can discover new things in the basin of things that have already discovered.
But, but we understand things very deep down in, in the epistemic tree, which means our, you know, our cone of creative potential is, is much wider.
1:00:03
It's possible that that gap might be closed.
What would happen then?
1:00:07
Speaker 1
The way I kind of think about the scientific process is like a tree search ultimately, right?
So I think a lot of sort of analogies from evolution transfer to scientific research, right, In the sense that we traverse a tree of different ideas or different experiments.
1:00:23
And then in the paper we report one path through that tree.
And I think what I kind of alluded to before, we need much more like full tree data sets for training these LLM systems to actually learn how to do this exploration and this foraging basically.
1:00:41
At the same time, I, I feel like evolution will also take place on the cultural level, like for us, right, we will get better at sort of steering the ship.
And I can't imagine that in, in a future world, sort of the way how we do research will be completely different.
1:00:56
And I'm pretty sure that right now already 99% of machine learning research is done with sort of AI assistance, right?
Think about ChatGPT brainstorming, cursor coding, cloud code, etcetera.
In the long run, we're going to move on that spectrum from sort of with AI, closer to by AI and then sort of more high level sort of orchestration and overseeing by humans.
1:01:20
Speaker 2
There's also the notion of how intrinsically coupled to humans is the value function.
So one school of thought is that AI will develop a mind of its own and it will, you know, basically transcend humanity and it will just have agency, which is not parasitic on on ours.
1:01:38
I personally don't subscribe to that view.
But the other view is that it is like, let's say the AI scientist, you know, like version 10.
It's going to be continually epistemic, you know, epistemic foraging.
It's going to be finding new things that are useful and they kind of have to be useful to us because if it finds things that are not useful to us, then we just won't use them and then nothing will happen.
1:02:01
So, so do do you think there'll always be a kind of coupled value function to humans?
1:02:05
Speaker 1
Jeff Kloon had this work on Omni, right, and using LLMS as sort of amortized notions of interestingness for humans, right?
And I think ultimately the way how we train these systems is coupled in, in human data, right?
And going forward, it will also be coupled with human data that is collected using verifiers, right?
1:02:25
So I have a hard time believing that in the long run, when you run this open endedness sort of paradigm with AI scientist agents, it's going to completely divert to, to something that's either fully non interpretable or unrelated to problems we as humans care about, right?
1:02:43
And then again, like humans can steer to a certain degree where like the search happens, right?
So you can tell the system, OK, try to do Cancer Research, right?
And sort of work on problems that we care about.
And ultimately, like we are the ones who control how much flops are being pushed into this.
1:03:01
Speaker 2
Yeah, because as a thought experiment, I can imagine, let's say, in the world of mathematics, what if an AI scientist could come up with entirely new problem formulations and then solve them?
1:03:11
AI Scientist V2: Agentic Tree Search and Falsificationism
And these are things that humans had never conceived of before.
And maybe they would be less interested in the answer because humans hadn't spent time thinking about it.
And if you think about it, we could just explore the phylogeny of mathematics just to the NTH degree, and at some point maybe we just wouldn't care anymore.
1:03:27
Maybe we can just carve out that space just forever and ever.
1:03:30
Speaker 1
Yeah.
But maybe down the road there is a stepping stone that enables a new innovation in a different field that we actually care about, right?
So it's very hard to say a priori whether or not something is interesting or not, right?
1:03:42
Speaker 2
Yes, and there's also the notion of I love this idea of diverse intelligences and diverse minds.
And maybe we, we could just create artifacts in a space which is completely alien to us and we might even ascribe moral value to them.
1:03:58
And we might not want to turn off, you know, the, the power because we, we want these alien artefacts to stay alive.
1:04:04
Speaker 1
Maybe like I'm, I read a lot of science fiction, but I would sort of shy away from from speculating about all of this.
But I do think one thing I'm extremely certain of is that the way how we conduct research and science is going to fundamentally change in the next 5 years, 10 years and 20 years.
1:04:24
And I hope that we're going to be able to sort of tackle some of the biggest problems which are still sort of seemingly unreachable right now with and by.
1:04:33
Speaker 2
AI so Terrance town has posted that he's been using GBT 5 to and and it's it's been speeding him up.
It's it's taking away a lot of the the drudgery.
But the cynical take is that, and Scott Harrison posted something similar as well.
1:04:48
The cynical take is that maybe laziness is stepping in and in some ernicious way using AI models is actually stopping us from thinking outside the box.
So it's it's encouraging us to kind of search in the neighborhood of things that are known.
1:05:04
And that is very useful.
It's very useful to have an artifact that knows all of the experiments, all of the things that are ever done by people 20 years ago.
But now we don't have people really kind of applying their their brilliance, their talent in completely new areas.
1:05:20
Speaker 1
So First off, it's great that these experts are already using the technology in their day-to-day work, right?
And I think it's also important that really, really top level scientists try to push what's capable with these systems or squeeze out where there might be sort of black spots or stuff where these systems can't do.
1:05:37
Second off, I think it comes down sort of to discipline and how we raise sort of the next generation, right?
So discipline on the personal level, like how much do you just sort of accept everything that's being proposed by these systems and responsibility in terms of educating the next generation in the sense that we need to sort of teach our kids that ultimately what comes out of these systems might not always be be true.
1:06:03
That facts can be sort of subjective, if you will, and that there needs to be more research about what's being given to you.
And I think this will be, like I said, this cultural evolution that we have to step through and try to make the best out of.
1:06:19
Speaker 2
Yeah, the autopilot thing is very interesting because there is a tendency using cursor just to, you know, at some point the models are getting so quickly that you can't even read, Yeah, the tokens coming at you.
And then you just press accept and you press accept it.
It's the same thing in cars that as soon as you have too strong of an autopilot, you just completely switch off and and then you see a divergent because there's something about thinking that it must be grounded on your path.
1:06:45
Addressing 'Slop' and Revolutionizing Scientific Publishing with AI
There's this path dependence and when you start kind of becoming parasitized by this other train of thought, then you stop thinking about your path and then you're not in the driver's seat anymore.
This.
1:06:55
Speaker 1
Is now like a bit of a harsh statement, but sometimes I wonder if these systems like these coding assistants are almost like drugs, right?
In the sense that you become addicted, you use up all your sort of budget and then you, you need to load up again.
1:07:10
And once you, you fully reach sort of the the budget limit, you feel like, OK, what am I going to do now?
And I think once that happens to you, you should really sort of rethink the way how you work, right?
And to me right now, there's certain parts where like sort of auto accepting is acceptable and there are certain parts where it's definitely not.
1:07:31
And you really need to go deep into it.
And I think right now we're sort of in this weird non equilibrium state where things are moving constantly, right?
So the systems or the models are changing, the features are changing.
The sort of points where the systems are good is change are changing all the time.
1:07:50
And we humans need to constantly adapt to to that, right?
And I think it's a big cognitive challenge.
And I think we just all need to be aware that there are certain problems and certain challenges that we have to adapt to.
1:08:05
I think the best way to do so is just interact with this technology as much as you can and maybe find new research ideas for out of that experience.
1:08:15
Speaker 2
And how is AI Scientist V2 different to V1?
1:08:18
Speaker 1
In V1 we we used sort of a template based approach.
So we had like a base experiment.
And then for that base experiment we asked sort of an LLM to generate ideas sort of with semantic scholar calls and sort of literature search.
And then it implemented sort of these ideas based on the template, right?
1:08:35
Basically code diffs and then it linearly executed like an experiment plan and wrote a paper in the end.
And So what could happen was that there was an idea and that idea didn't work out right.
But then in the end, the paper, like the experiments were still executed linearly and you wrote a paper.
1:08:52
And this was already impressive in the sense that it looked very much like like science.
But if you think about human sort of science and like the scientific method, it's much more like research, like I said before, right?
You sort of adapt what you're going to execute next and you sort of refine based on evidence that you accumulated, right?
1:09:12
So this is sort of the, the notion of falsificationism from, from Karl Popper, right?
In the sense that we collect evidence for hypotheses and reject, we reject others.
And we do so in, in a loop basically until we we want to publish or we find something.
And we try to take this notion and directly build it into the agentic scaffolding for the AI scientist V2.
1:09:34
So now it's basically like an paralyzable agentic tree search where there's no longer a template experiment needed.
But this is drafted up by the LLM itself and thereby the AI scientist V2 can be applied to many more sort of settings, if you will.
So at the core is sort of this new agentic tree search paradigm.
1:09:53
And then we use sort of a couple of minor technical chain changes like using a VLM reviewer for sort of figuring out if captions of a paper are aligned with the figures.
And we we scale this up to many more sort of computational nodes and then write a paper in the end again.
1:10:11
Speaker 2
So I'm trying to say this in the most polite way possible, but a critic might say, I don't want to use the word slot, but a critic might say we are producing papers which appear like papers so that they have figures and they have results and they have things written in a certain way, but they're not grounded deep down the epistemic phylogeny, which means that they, they, they have, you know, near, near the top of the tree.
1:10:36
We're seeing some novelty and composition happening, but it, but it doesn't reflect a deep understanding.
What would you say to that charge?
1:10:43
Speaker 1
It's for sure that not every paper that comes out of the AI Scientist V2 is a nature worthy publication, right?
That that's for sure the case.
So definitely there is some amount of, let's say, slop or content that is not like a scientific big discovery being written up by the AI Scientist V2.
1:11:03
But ultimately, like we, we showed that it was possible to obtain a workshop level paper.
I do think this is sort of the first time basically where we can see that at least now we're able to fully autonomously spend compute, spend API calls to obtain some amount of scientific insights.
1:11:19
And for me, at least right now, it's a good way to sort of prototype ideas or to investigate a certain field, get like initial starting point, initial results, and then to to work on top of it.
But for sure, more work needs to be done to make this entire process more robust, more efficient, and essentially produce many more sort of true positives as you will.
1:11:41
Speaker 2
Yeah, and it might be one of these things, you know, like when we moved from GPT 3 to GPT 4, there was just a massive increase in fidelity is The thing is with with slop, to me it simply means lack of deep grounded understanding.
And there's no reason in principle why these things couldn't have a deep grounded understanding.
1:11:58
They just don't have it yet.
So it's something that could improve over time, but it's likely to improve quite slowly.
And then at some point we might just think, Oh my God, we've got an AI scientist.
1:12:08
Speaker 1
Yeah.
I mean, like to me this this kind of comes back to what we were discussing about before.
So First off, there is a verifier in the loop, right?
Or in the sense that experiments are actually executed on a computer, right.
So the numerical results can be be fed back or are fed back into the system to come up with the next thing to explore.
1:12:29
But like we haven't made like a, let's say discovery like residual connections or something I think that have diffused into everything in machine learning.
And I think what we really need is to make these systems be much better at sort of integrating knowledge over multiple experiments and sort of become better at sort of formulating the next hypothesis based on previous insights.
1:12:52
And yeah, this might require some amount of post training on sort of these traces basically.
But I'm pretty positive that we might also get there with just diversity and scaling these systems up in in an efficient but as get up way.
1:13:08
Speaker 2
I'm just thinking that the the first breakthrough discovery, would it resemble the AOI scientist paper or would it resemble Chinka revolve?
So for example, we, we could do like a massively scaled up Chinka revolve and we could say, I want to discover a new architectural design and would that happen?
1:13:25
And then we would get the AI scientist paper to kind of write it up and do ablations and stuff.
Maybe that would be the the pattern of it?
1:13:32
Speaker 1
To a certain degree, I've been thinking a lot about how you can potentially even combine these two paradigms, right?
The AI scientist and and and Chinka or alpha evolve style optimization algorithms.
And I do think there is some amount of work to be done on sort of this auto verification sort of aspect to it on the sort of problem formulation aspect to it.
1:13:52
The paper writing part is actually the least important about the AI scientist, right?
It's a form factor that we humans are sort of used to and it helps anchor our mental model of like a scientific discovery.
But ultimately I'm not sure if the paper is going to be the the knowledge transmission medium in let's say 20 years, right?
1:14:15
Something else I've been thinking of a lot is whether or not we can make papers much easier and genetically accessible, right?
In the sense that right now it's it's it's a Latex document.
But you could imagine sort of equipping every paper with sort of several model context protocols so that every figure is reproducible, data is accessible, and essentially make it much easier for the LLM agents to essentially either replicate work or to work off of them afterwards, right?
1:14:47
Doing sort of epsilon improvements ablations yourself through that interface to a paper.
But to be entirely honest, I'm not sure if it's going to happen because there have been many great ideas for improving sort of, let's say the, the format of scientific artifacts out there.
1:15:04
And people still seem to to like the paper format, which has existed for let's say hundreds of years, right?
So I think it's a question of incentives again and really showing that if something like that would exist, it would enable much faster progress of AI agents for scientific discovery.
1:15:23
Speaker 2
Yeah, paper is a great human interface.
It's a similar thing with automated driving, right?
That we could revolutionize the road network to have sensors and we could dramatically improve the the monitoring and observability and optimization.
But I'm fascinated by that idea.
1:15:38
So, so you're saying not just reproducibility of the experiments, but also the way that the figures are designed and, and the code and and so on, Because then we could create this huge playground where agents can repurpose, recombine, re study work that has been published by other scientists.
1:15:55
And it also made me think, does like having an automated scientist, does that make peer review more or less important?
1:16:02
Speaker 1
I do think it's actually makes it more important, at least for now, right?
In the sense that we now have a mechanism or could have a mechanism that generates many, many papers, right?
And it first increases like the workload on on on human reviewers and we need some effective way for filtering and then essentially only taking the cream of the crop for human verification afterwards, right.
1:16:26
So I think for now, like the ultimate verification is still like the human and the diffusion of the result through the community.
And we need better tools for doing this automatic filtering and verification.
Like we have the AI reviewer that comes with sort of the AI scientist, but you actually probably need some form of experiment execution for actually verifying everything.
1:16:50
Yeah, but there is, for example, work by open AI on on paper bench and trying to go into that direction using sort of LLM soft verification and these types of things.
So I I'm hopeful that we're going to figure this out in the next years.
1:17:04
Speaker 2
Yeah, and I think one of the Rubicon moments is when the the new Transformers architectural, something massive is discovered by AI and we're all using it.
My worry, I suppose, is that probably folks like Google who have enough compute power, they're going to be running AI scientists and they're going to own many of these discoveries, which is why it's so important to have work which can efficiently discover new things in science.
1:17:28
Speaker 1
And it's important to have work that's openly available, right?
I think like with the AI scientist and Shinka, we're really trying to to make sure that we can sort of apply the collective intelligence of all of us to to shape how this might look in the future.
1:17:42
Speaker 2
Amazing.
Rob, this has been so fantastic to have you on the show.
Sakana is hiring amazing engineers by the way.
So if if this sounds like and it, it is an amazing opportunity, get in touch with Robin and the guys and I trust you're working on some exciting new things that are coming up.
1:17:57
Speaker 1
Yes.
And I, I hope to be able to talk to you in the future again about some of this.
1:18:01
Speaker 2
Absolutely, Rob, thank you so much for coming.
1:18:02
Speaker 1
Thank you so much.
1:18:03
Speaker 2
Tim.