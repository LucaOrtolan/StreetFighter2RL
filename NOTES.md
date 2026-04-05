[More extensive notes can be found here](https://maynoothuniversity-my.sharepoint.com/:w:/r/personal/krystal_davis_2026_mumail_ie/_layouts/15/Doc.aspx?sourcedoc=%7BC6F86477-7BFB-42C4-96B6-79DB6CA9822D%7D&file=Document.docx&action=editNew&mobileredirect=true&wdOrigin=EXCELONLINE.SHELL%2CAPPHOME-WEB.UNAUTH%2CAPPHOME-WEB.SHELL.SIGNIN%2CAPPHOME-WEB.BANNER.NEWBLANK&wdPreviousSession=a40f00b0-51b3-4f5d-b3a8-fd1d4034387b&wdPreviousSessionSrc=AppHomeWeb&ct=1775298109638)

# Using NN to predict damage
- First, the Aux Head NN learns to approximate damage.
- Then, as an agent gets better at not taking damage, the aux NN gets worse at predicting it and is not really helpful

# New Trials
- Trying to capture the need for blocking with damage_taken mapping. 3 experiments:
  - Note: A coeffecient/modifier is created for damage taken to determine the "strength" of the penalty, based on the max damage taken in the round.
    - Exp 1: Penalty for damage taken is calibrated/tuned to give lower relative penalties for lower damage amounts.
    - Exp 2: Higher penalties for relatively higher damage amounts.
    - Exp 3: Design a sigmoid : ¿por qué no los dos?
- Stable-Retro3 default CNN not optimized.
  - Optimization is finished. (See paranoia.txt)
