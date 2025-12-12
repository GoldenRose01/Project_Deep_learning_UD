from deep_translator import GoogleTranslator


class TranslationService:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')

    def translate_to_en(self, text):
        try:
            return self.translator.translate(text)
        except:
            return text

    def translate_from_en(self, text, lang_code):
        try:
            return GoogleTranslator(source='en', target=lang_code).translate(text[:2000])
        except:
            return text