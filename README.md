# dragonfly

A "System One" decision model for images. You give it one image and many typed questions, and it answers all of them in a single forward pass. Each answer is a calibrated probability distribution; the model never generates text.

The idea comes from TypeSafe AI's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), which does this for text. dragonfly applies it to images.

## Question types

| Type | Example | Output |
|---|---|---|
| Bool | "Is there a car?" | P(yes), P(N/A) |
| Choice | "Time of day? [morning, afternoon, night]" | Distribution over options, P(N/A) |
| Score | "Damage, 1 = pristine … 10 = totaled" | Distribution over the scale, P(N/A) |

## Approach

- A frozen vision encoder feeds a LoRA-tuned VLM.
- All questions are packed into one sequence behind the image. An isolation mask and shared position ids keep each question independent of the others and of their order.
- Typed heads read each question's `[DECIDE]` and `[OPT]` tokens. There's no LM head and no decoding loop, which is where the speed comes from.
- Training uses proper scoring rules, followed by temperature scaling for each head type.

## What it should show

1. Latency stays nearly flat as the number of questions grows. Autoregressive answering grows steeply.
2. Its reliability diagram sits close to the diagonal, meaning its confidence matches its accuracy.
3. Shuffling the questions or the options changes the outputs by exactly 0.
4. On question types held out of training, it scores at least as well as the untrained baseline.

## Status

Planning. No code yet.
