import pandas as pd
import re
import unicodedata
import langid

# Optionnel mais très utile pour corriger les titres mal encodés
try:
    from ftfy import fix_text
    USE_FTFY = True
except ImportError:
    USE_FTFY = False

# =========================
# 1) Chargement
# =========================
input_path = r"C:\Users\tedvm\Centrale\PoleProjetS8\Data_PublicHealth_CentraleSupelec\data\media_cloud\mediacloud_urls.csv"

df = pd.read_csv(input_path)

# =========================
# 2) Restriction des langues testées
# =========================
langid.set_languages(['fr', 'en', 'es', 'de', 'it'])

# =========================
# 3) Nettoyage des titres
# =========================
def clean_title(title):
    if pd.isna(title):
        return ""

    title = str(title)

    # Corrige les erreurs du type Iâ€™m -> I’m
    if USE_FTFY:
        title = fix_text(title)

    # Normalisation unicode
    title = unicodedata.normalize("NFKC", title)

    # Suppression / remplacement de caractères spéciaux
    title = title.replace("\xa0", " ")
    title = title.replace("\u200b", "")

    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for old, new in replacements.items():
        title = title.replace(old, new)

    # Réduction des espaces multiples
    title = re.sub(r"\s+", " ", title).strip()

    return title

# =========================
# 4) Détection de langue
# =========================
def detect_language(title):
    if pd.isna(title) or not str(title).strip():
        return None, None

    lang, score = langid.classify(str(title))
    return lang, score

# =========================
# 5) Détection des titres trop courts
# =========================
def is_title_too_short(title, min_words=3, min_letters=15):
    if pd.isna(title) or not str(title).strip():
        return True

    title = str(title)
    words = re.findall(r"\b\w+\b", title)
    letters = re.findall(r"[A-Za-zÀ-ÿ]", title)

    return len(words) < min_words or len(letters) < min_letters

# =========================
# 6) Heuristique simple FR/EN
#    pour aider sur certains cas ambigus
# =========================
FRENCH_HINTS = {
    "le", "la", "les", "des", "du", "de", "un", "une", "et", "pour", "avec",
    "sur", "dans", "apres", "après", "contre", "par", "au", "aux", "en",
    "que", "qui", "est", "sont", "plus", "mois", "annee", "année"
}

ENGLISH_HINTS = {
    "the", "and", "for", "with", "why", "off", "month", "from", "after",
    "against", "is", "are", "new", "how", "what", "this", "that", "takes",
    "taking", "years"
}

def count_language_hints(title):
    if not title:
        return 0, 0

    words = re.findall(r"\b[a-zà-ÿA-ZÀ-ÿ']+\b", str(title).lower())
    fr_count = sum(word in FRENCH_HINTS for word in words)
    en_count = sum(word in ENGLISH_HINTS for word in words)

    return fr_count, en_count

# =========================
# 7) Classification finale
# =========================
def classify_title_language(row):
    title = row["title_clean"]
    lang = row["lang"]
    short = row["short_title"]
    fr_hints = row["fr_hints"]
    en_hints = row["en_hints"]

    # Titre vide
    if not title:
        return "uncertain"

    # Si le titre est court, on évite de trancher trop vite
    if short:
        # on tranche seulement si les indices sont très clairs
        if en_hints >= 2 and fr_hints == 0:
            return "non_fr"
        elif fr_hints >= 2 and en_hints == 0 and lang == "fr":
            return "fr"
        else:
            return "uncertain"

    # Cas normaux
    if lang == "fr":
        # si beaucoup d'indices anglais malgré détection fr, on préfère incertain
        if en_hints >= 2 and fr_hints == 0:
            return "uncertain"
        return "fr"

    if lang == "en":
        # si beaucoup d'indices français malgré détection en, on préfère incertain
        if fr_hints >= 2 and en_hints == 0:
            return "uncertain"
        return "non_fr"

    # Autres langues
    if lang in {"es", "de", "it"}:
        return "non_fr"

    return "uncertain"

# =========================
# 8) Application au dataframe
# =========================
df["title_clean"] = df["title"].apply(clean_title)

df[["lang", "lang_score"]] = df["title_clean"].apply(
    lambda x: pd.Series(detect_language(x))
)

df["short_title"] = df["title_clean"].apply(is_title_too_short)

df[["fr_hints", "en_hints"]] = df["title_clean"].apply(
    lambda x: pd.Series(count_language_hints(x))
)

df["lang_class"] = df.apply(classify_title_language, axis=1)

# =========================
# 9) Sous-dataframes
# =========================
df_fr = df[df["lang_class"] == "fr"].copy()
df_uncertain = df[df["lang_class"] == "uncertain"].copy()
df_non_fr = df[df["lang_class"] == "non_fr"].copy()

# =========================
# 10) Export
# =========================
base_dir = r"C:\Users\tedvm\Centrale\PoleProjetS8\Data_PublicHealth_CentraleSupelec\data\media_cloud"

df_fr.to_csv(fr"{base_dir}\mediacloud_urls_fr.csv", index=False, encoding="utf-8-sig")

# =========================
# 11) Petit résumé
# =========================
print("Nombre total de lignes :", len(df))
print("Articles français :", len(df_fr))
print("Articles non français :", len(df_non_fr))

print("\nExemples FR :")
print(df_fr[["title", "title_clean", "lang", "lang_score"]].head(10))