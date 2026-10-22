"""pytest: per-profile isolation, retrieval, dedup, rules, model-swap re-embed. CPU only, tmp dir."""
import sys, pathlib
import pytest

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


@pytest.fixture()
def store_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    import importlib, memory_store
    importlib.reload(memory_store)
    memory_store.MEMORY_DIR = tmp_path
    memory_store._stores.clear()
    return memory_store


FACTS_A = ["User's name is Alex.", "User's cat is called Whiskers.", "User has a team meeting every Thursday at 8 pm.",
           "User has a project deadline on 14 November.", "User drinks green tea."]
FACTS_B = ["User's name is Sam.", "User has a maths exam on Friday.", "User's study partner is Chris."]


def _fill(store, facts):
    for f in facts:
        store.add(f)
    return store


def test_profiles_are_separate_files_and_contexts(store_mod, tmp_path):
    a = _fill(store_mod.get("alice"), FACTS_A)
    b = _fill(store_mod.get("bob"), FACTS_B)
    store_mod.get("nova")
    assert {p.name for p in tmp_path.glob("*.sqlite")} == {"alice.sqlite", "bob.sqlite", "nova.sqlite"}
    assert a.count() == len(FACTS_A) and b.count() == len(FACTS_B)
    assert store_mod.get("nova").count() == len(store_mod.PROFILES["nova"]["facts"]) == 0  # ships with no facts
    f, ms = a.retrieve("what is my cat's name")
    assert f and "Whiskers" in f[0]["text"]
    assert ms < 50
    f2, _ = b.retrieve("what is my cat's name")
    assert all("Whiskers" not in x["text"] for x in f2), "no cross-profile leakage"
    f3, _ = b.retrieve("when is my exam")
    assert f3 and "exam" in f3[0]["text"].lower()
    a.add("User's dog is called Luna.")
    assert "Luna" not in " ".join(x["text"] for x in b.list())
    assert store_mod.get("alice").count() == len(FACTS_A) + 1


def test_unrelated_query_returns_nothing(store_mod):
    f, _ = _fill(store_mod.get("alice"), FACTS_A).retrieve("what is the speed of light")
    assert f == []


def test_dedup_updates_instead_of_inserting(store_mod):
    s = _fill(store_mod.get("bob"), FACTS_B)
    n = s.count()
    a = s.add("User's study partner is Chris, a friend.")
    assert s.count() == n and a is not None  # near-duplicate → update
    b = s.add("User's favourite colour is green.")
    assert s.count() == n + 1 and b != a


def test_delete_and_reset(store_mod):
    s = store_mod.get("nova")
    fid = s.add("User's cat is called Mani.")
    assert s.delete(fid) and s.count() == 0
    s.add("User owns a scooter.")
    assert s.reset() == 0  # back to the (empty) seed set
    assert s.count() == 0


def test_rules_extraction(store_mod):
    ex = store_mod.extract_facts
    assert ex("my dog is called Bruno") == ["User's dog is called Bruno."]
    assert ex("My sister's name is Leela.") == ["User's sister is called Leela."]
    assert ex("my dog's called Bruno") == ["User's dog is called Bruno."]
    assert ex("my favourite colour is green") == ["User's favourite colour is green."]
    assert ex("my mother's birthday is on 3 May") == ["User's mother's birthday is on 3 May."]
    assert ex("I have it") == []
    assert "User asked to be reminded to call the dentist tomorrow." in ex("please remind me to call the dentist tomorrow")
    assert ex("I love pancakes, they're the best") == ["User loves pancakes."]
    assert ex("I'm allergic to peanuts") == ["User is allergic to peanuts."]
    assert ex("I'm going to Lisbon next month") == ["User is going to Lisbon next month."]
    assert ex("how long do I boil eggs") == []
    assert ex("what have I got on this week") == []
    assert ex("what is the name of my dog") == []
    assert ex("do I have anything on tomorrow") == []
    assert ex("I have a dentist appointment on Monday. What is a black hole?") == ["User has dentist appointment on Monday."]
    assert ex("what is a black hole") == []


def test_compose_prompt_budget(store_mod):
    f, _ = _fill(store_mod.get("alice"), FACTS_A).retrieve("what have I got this week with Mira and Mum")
    p = store_mod.compose_prompt("what have I got this week", f)
    assert p.startswith("Things you remember") and "The person says: what have I got this week" in p
    assert sum(len(x["text"].split()) for x in f) <= store_mod.BUDGET_WORDS


def test_model_swap_reembeds(store_mod, monkeypatch):
    s = _fill(store_mod.get("alice"), FACTS_A)
    s.db.execute("UPDATE facts SET embed_model='old-model', dim=384")
    s.db.commit()
    s2 = store_mod.MemoryStore("alice", path=s.path)
    assert s2.db.execute("SELECT count(*) FROM facts WHERE embed_model='old-model'").fetchone()[0] == 0
    f, _ = s2.retrieve("what's my cat's name")
    assert f and "Whiskers" in f[0]["text"]


def test_schedule_questions_pull_dated_facts(store_mod):
    a = _fill(store_mod.get("alice"), FACTS_A)
    f, _ = a.retrieve("What have I got on this week?")
    texts = " ".join(x["text"] for x in f)
    assert "Thursday" in texts and "November" in texts
    assert all(x["why"] in ("schedule", "similar") for x in f)
    f2, _ = a.retrieve("what is a black hole")
    assert f2 == []
    b = _fill(store_mod.get("bob"), FACTS_B)
    f3, _ = b.retrieve("anything on tomorrow?")
    assert any("exam" in x["text"].lower() for x in f3)
    assert all("Whiskers" not in x["text"] for x in f3)


def test_loose_phrasings_are_remembered():
    from memory_store import extract_facts as ef
    assert ef("By the way, Leela is my sister.") == ["User's sister is called Leela."]
    assert ef("We got a new puppy, his name is Bruno.") == ["User has a puppy called Bruno."]
    assert ef("I moved to Denver last month.") == ["User moved to Denver last month."]
    assert ef("I can't eat peanuts, I'm allergic.") == ["User is allergic to peanuts."]
    assert ef("I started working at Acme.") == ["User works at Acme."]
    assert ef("This is my sister.") == [] and ef("What is my sister called?") == []
