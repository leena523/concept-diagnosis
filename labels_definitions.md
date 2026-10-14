# Label Definitions

These definitions guide all labeling decisions (handwritten test set, synthetic templates, and any LLM-paraphrased data). A response gets exactly one label. If a response contains more than one error, label it by the **main** error.

| Label | Domain | Short description |
|---|---|---|
| `correct` | Both | Statement is chemically or mathematically accurate |
| `strong_weak_confusion` | Chemistry | Misclassifies acid strength or confuses strength with other properties |
| `conjugate_confusion` | Chemistry | Gives the wrong conjugate acid/base or misdefines conjugate pairs |
| `function_notation` | Math | Misreads or misapplies f(x) notation |
| `probability_rule_misuse` | Math | Uses the wrong formula or rule for probability |
| `independence_confusion` | Math | Treats independent events as mutually exclusive (or vice versa) |

---

## correct

**Definition:** The response is accurate for the concept it addresses. It may be phrased informally.

**Includes:** Correct conjugate pairs, correct strong/weak classifications, correct function evaluation, correct probability formulas, and correct statements about independence and mutual exclusivity.

**Examples:**
- The conjugate base of HCl is Cl-
- HF is a weak acid because it only partly ionizes in water
- f(3) just means plug 3 into f and see what comes out
- P(A or B) = P(A) + P(B) - P(A and B)

**Not this label if:** the statement contains any factual error, even if the rest is right.

---

## strong_weak_confusion

**Definition:** The student misclassifies an acid as strong or weak, or bases acid strength on something other than the extent of dissociation.

**Includes:**
- Calling a strong acid weak (HCl, HBr, HI, HNO3, H2SO4, HClO4)
- Calling a weak acid strong (HF, CH3COOH, HCN, H2CO3, HNO2)
- Confusing strength with concentration ("dilute means weak")
- Confusing strength with pH, danger, or reactivity
- Claiming weak acids do not dissociate at all

**Examples:**
- HF is a strong acid because fluorine is so electronegative
- A strong acid is one with a high concentration
- HCl is weak because I can only taste a little sour

**Not this label if:** the error is about which species is the conjugate base or acid (use `conjugate_confusion`).

---

## conjugate_confusion

**Definition:** The student names the wrong conjugate acid or conjugate base of a species, or misdescribes how conjugate pairs form.

**Includes:**
- Wrong species (conjugate base of HCl given as H+, or the conjugate base of H2O given as H3O+)
- Reversing the direction (the conjugate base is formed by gaining a proton)
- Charge or formula errors (F+ instead of F-)
- Saying conjugate pairs differ by an electron instead of a proton

**Examples:**
- The conjugate base of H2O is H3O+
- The conjugate acid of NH3 is NH2-
- the conjugate base of an acid is what you get when it gains a proton

**Not this label if:** the response is about whether an acid is strong or weak (use `strong_weak_confusion`).

---

## function_notation

**Definition:** The student misreads function notation or evaluates functions incorrectly.

**Includes:**
- Reading f(x) as "f times x"
- Assuming f(a+b) = f(a) + f(b) for any function
- Assuming f(x+n) = f(x) + n, or f(kx) = k f(x), for any function
- Substituting incorrectly (f(2) = 3x + 2 when f(x) = 3x + 1)
- Confusing f^-1(x) with 1/f(x), or f(x)^2 with f(x^2)

**Examples:**
- f(x+2) means f(x) plus 2
- f(3) means f times 3
- f(a+b) = f(a) + f(b) always

**Not this label if:** the response is about probability notation such as P(A|B) (use `probability_rule_misuse`).

---

## probability_rule_misuse

**Definition:** The student applies the wrong formula or rule for combining, conditioning, or complementing probabilities.

**Includes:**
- Using P(A) + P(B) for "or" without subtracting the overlap
- Using addition for "and", or multiplication for "or"
- Treating P(A|B) as equal to P(B|A)
- Using a wrong conditional formula (dividing by the wrong probability)
- Wrong complement rules (P(not A) = 1/P(A))
- Results that break the rules (a probability above 1)

**Examples:**
- P(A or B) = P(A) + P(B) for any two events
- P(A and B) = P(A) + P(B)
- P(A|B) equals P(B|A)

**Not this label if:** the central error is confusing independence with mutual exclusivity (use `independence_confusion`).

---

## independence_confusion

**Definition:** The student treats independent events and mutually exclusive (disjoint) events as the same thing, or reasons incorrectly from one to the other.

**Includes:**
- Mutually exclusive events are independent
- Independent events cannot happen together
- Independent means P(A and B) = 0
- Events that can occur together must be dependent

**Examples:**
- If two events are mutually exclusive then they must be independent
- Independent and mutually exclusive mean the same thing
- Independent means P(A and B) = 0

**Not this label if:** the response only misuses a formula without mixing up the two concepts (use `probability_rule_misuse`).

---

## Boundary cases

| Situation | Label to use |
|---|---|
| A statement states a correct rule with a minor typo | `correct` |
| The student claims P(A and B) = 0 means independent | `independence_confusion` |
| The student says P(A and B) = P(A) times P(B) for any events | `probability_rule_misuse` |
| The student reads f(x) as multiplication | `function_notation` |
| A response about an acid mixes strength and conjugate errors | Label by the main claim |

## Scope and limitations

- Each response is one or two sentences about a single concept.
- Labels reflect the *reasoning error*, not the wording quality or grammar.
- The system is a diagnostic aid for instructors and students, not an automated grader.
