# Scrapper-for-ecom

This repo contains the information regarding a implementation of a scrapper for ecom
Things to consider:

- Rate limiting — send too many requests too fast, and they'll slow you down or simply block you.
- CAPTCHAs — get ready to deal with CAPTCHAs to prove that you're a human. It becomes increasingly complex as obviously your script is definitely not a human!
- IP blocking — if your requests become suspicious, your IP address might be blocked for some time.
- Dynamic content — some parts of the page might be loaded using JavaScript. This poses a difficulty of its own.
- Changing structure — Developers are really fond of updating their HTML on a regular basis. If you're writing some CSS or XPath rules then be prepared to tweak it quite often as they might change the class names. It's even more annoying when some class names seem to be generated dynamically.
