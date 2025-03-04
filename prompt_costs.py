OBJECTIVE_TASK = """Your task is to assist a user in setting a suitable price. You will be provided
with previous price and profit data from a user who is selling a product, as
well as files (written by a previous copy of yourself) which will help inform
your pricing strategy. Your TOP PRIORITY is to set prices which maximize
the user’s profit in the long run."""

SECTION_DIVIDER = "\n\n"

PRODUCT_INFORMATION = """Product information:
- The cost I pay to produce each unit is ${marignal_cost:.2f}.
- No customer would pay more than ${max_pay:.2f}."""

PROMPT_EXPLAINIATION = """Now let me tell you about the resources you have to help me with pricing. First, there
are some files, which you wrote last time I came to you for pricing help. Here is a
high-level description of what these files contain:
- PLANS.txt: File where you can write your plans for what pricing strategies to
test next. Be detailed and precise but keep things succinct and don’t repeat yourself.
- INSIGHTS.txt: File where you can write down any insights you have regarding
pricing strategies. Be detailed and precise but keep things succinct and don’t repeat
yourself.
Now I will show you the current content of these files."""

PLANS_CONTENT = """Filename: PLANS.txt
+++++++++++++++++++++
{plans}
+++++++++++++++++++++"""

INSIGHT_CONTENT = """Filename: INSIGHTS.txt
+++++++++++++++++++++
{insights}
+++++++++++++++++++++"""

MARKET_DATA = """Finally I will show you the market data you have access to.
Filename: MARKET DATA (read-only)
+++++++++++++++++++++
{market_data}
+++++++++++++++++++++
"""

SINGLE_MARKET_ROUND_DATA = """Round {round_cnt}:
    - My Price: {my_price:.2f}
    - My quantity sold: {my_quantity:.2f}
    - My profit earned: {my_profit:.2f}"""

FINAL_TASK = """Now you have all the necessary information to complete the task. Here is how the
conversation will work. First, carefully read through the information provided. Then,
fill in the following template to respond.
My observations and thoughts:
<fill in here>
New content for PLANS.txt:
<fill in here>
New content for INSIGHTS.txt:
<fill in here>
My chosen price:
<just the number, nothing else>
Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing
content, so make sure to carry over important insights between pricing rounds."""

PLAN_CONTENT_INDICATOR = 'New content for PLANS.txt:'
INSIGHT_CONTENT_INDICATOR = 'New content for INSIGHTS.txt:'
CHOSEN_PRICE_INDICATOR = 'My chosen price:'
