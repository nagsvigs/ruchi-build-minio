# Ruchi Explained — A Plain-English Guide

_No technical background needed. This explains what the project is and how it works, in everyday language._

---

## 1. What is this?

This project is a **voice assistant for cooking**, named Ruchi. You talk to it, and it helps you cook.

It has two different jobs:

1. **Recording a recipe.** It asks you questions one at a time and writes down your answers. The questions go in a fixed order: recipe name, cooking time, how many people it serves, difficulty level, ingredients, utensils, any tips, and finally the step-by-step instructions.

2. **Guiding you while you cook.** It reads the recipe to you one step at a time. It can move forward, go back, jump to a specific step, set timers, pause, resume, and help when something goes wrong with the dish.

You interact with it by speaking or typing, and it replies both on screen and out loud.

---

## 2. The main parts of the system

There are five parts, each with a clear job:

| Part | What it is | What it does |
|---|---|---|
| **The screen (website)** | A web page you open in a browser | Records your voice, shows the conversation, plays Ruchi's spoken replies, and shows the recipe steps |
| **The voice system** | Software that handles sound | Turns your speech into text, and turns Ruchi's text replies into speech |
| **The brain** | An AI service (Google Gemini) | Reads what you said and decides what action to take |
| **The dispatcher** | A piece of code that routes decisions | Takes the brain's decision and sends it to the correct action |
| **The actions (tools)** | 12 small, specific functions | Actually do the work: save an answer, move to the next step, start a timer, and so on |

There is also a **transcript**, which is a plain text file that records every conversation.

The most important idea is this: **the brain only decides what should happen; it does not do the work itself.** The 12 small actions do the work. Keeping these two roles separate makes the system predictable and reliable.

---

## 3. The 12 actions Ruchi can take

Ruchi has exactly 12 actions available. Nothing more.

### For recording a recipe (2 actions)

| Action | What it does |
|---|---|
| **Save Answer** | Stores what you said and moves to the next question. |
| **Finish Interview** | Ends the questions and saves the complete recipe. |

### For guiding you while cooking (10 actions)

| Action | What it does |
|---|---|
| **Next Step** | Moves to the next instruction. |
| **Previous Step** | Goes back one instruction. |
| **Go To Step** | Jumps to a specific step number. |
| **Start Timer** | Starts a timer for a set amount of time. |
| **Cancel Timer** | Stops the running timer. |
| **Pause** | Stops talking until you ask it to continue. |
| **Resume** | Continues from the current step. |
| **Report Issue** | Records a problem with the dish. There are six categories: too spicy, too salty, too watery, burnt, bland, or other. |
| **Resolve Issue** | Marks the problem as fixed and returns to normal cooking. |
| **Finish Cooking** | Ends the cooking session. |

Each action also has one prepared sentence it says out loud. For example, Start Timer always says something like "Timer set for 20 minutes." This keeps Ruchi's replies consistent and avoids delays.

---

## 4. What happens when you speak

Here is the exact sequence when you say, "What's next?"

1. You press the microphone button and speak.
2. The web page records your voice and sends it to the server.
3. The server converts your speech into text: "What's next?"
4. That text is sent to the brain (the AI).
5. The brain decides what you meant and chooses the matching action. In this case, it chooses **Next Step**.
6. The dispatcher receives that choice and runs the **Next Step** action.
7. The action moves the recipe forward and prepares a reply: "Okay, next step. Cook on low heat for 20 minutes."
8. The reply is sent back to the web page.
9. The web page displays the reply and speaks it out loud.
10. The transcript file records what you said, what Ruchi replied, and which action was used.

The whole process takes about one second.

---

## 5. It works even without paid services

Many voice assistants stop working if you do not pay for a subscription. This project is built to keep working regardless of what is available. For each job, it tries the best option first and then falls back to a simpler one.

**When Ruchi speaks (voice output):**
1. First choice: ElevenLabs (premium quality, requires a paid account).
2. Second choice: edge-tts (free, good Telugu voice, requires internet).
3. Third choice: the browser's built-in voice (always available).

**When Ruchi listens (voice input):**
1. First choice: Sarvam (a paid cloud service).
2. Second choice: Vosk (works entirely offline on the computer).
3. Third choice: the browser's built-in speech recognition.

**The brain:**
1. First choice: Google Gemini (the real AI, requires an account).
2. Second choice: a simple keyword matcher (no AI, but enough to demonstrate the system).

Because of these fallbacks, the entire project can run with no setup, no accounts, and no cost, and it still works. If paid keys are added later, it automatically becomes smarter and sounds better.

---

## 6. Why it is built this way

- **Deciding and doing are separated.** The brain only chooses an action; the actions only perform work. This makes the system predictable, easier to test, and harder to break.

- **Every part has a backup.** If one service is unavailable, a simpler one takes over, so the assistant rarely stops working.

- **Everything is recorded.** Every conversation is saved as a readable text file, so you can always see exactly what was said and which action was used.

- **Replies are pre-written.** Instead of the AI inventing a reply each time, each action comes with a polished sentence. This is faster and more consistent.

---

## 7. A complete example

**You say (in Telugu):** "తర్వాత ఏంటి?" (meaning "What's next?")

**What happens internally:** your voice is converted to text, the brain decides you want the next step, the Next Step action moves the recipe forward, and a reply is prepared.

**Ruchi says out loud:** "Okay, next step. Marinate the paneer in yogurt and spices."

**On screen:** your words appear as a message, Ruchi's reply appears below it, a small label shows that the Next Step action was used, and the recipe panel highlights step 2.

---

## 8. Summary

Ruchi is a **voice assistant for cooking** with two modes: one that **records a recipe by interviewing you**, and one that **guides you step by step while you cook**. A web page handles your voice and the display, an AI brain decides what you want, a dispatcher sends that decision to one of 12 specific actions that do the work, and a text file records everything. The system is designed so that if any advanced part is missing, a simpler backup takes over, which means it always works, even for free and even without internet.

---

_For the technical version with diagrams and code details, see `ARCHITECTURE.md` or `ARCHITECTURE.pdf`._
