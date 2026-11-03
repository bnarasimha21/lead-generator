"""Adversarial-review finding 1: evidence being a substring of the
transcript is not enough. The value asserted for a field must itself be
derivable from its own evidence quote, not just "somewhere in the
transcript". These are the exact probes from the review.
"""

from app.schema import EvidencedField, Lead
from app.verify import value_supported_by_evidence, verify_lead


def _lead_with(**fields):
    lead = Lead.empty()
    for name, (value, evidence) in fields.items():
        setattr(lead, name, EvidencedField(value=value, evidence=evidence))
    return lead


def test_value_with_unrelated_evidence_in_transcript_is_rejected():
    """The exact probe: transcript mentions a real name/company/email, but
    the record claims email=attacker@evil.example with evidence="my email
    is" (a real substring of the transcript, but no actual email address
    anywhere in that span). Must be rejected.
    """
    transcript = (
        "Hi, I'm Dana from Initech. My email is priya at acme dot io, "
        "let's follow up next week."
    )
    lead = _lead_with(email=("attacker@evil.example", "my email is"))
    verified, rejected = verify_lead(lead, transcript)
    assert "email" in rejected
    assert verified.email.value is None
    assert verified.email.evidence is None


def test_value_asserted_against_unrelated_one_token_evidence_is_rejected():
    """Same failure mode with a different field: a one-token evidence
    string that is a real substring of the transcript, but shares nothing
    with the asserted value.
    """
    transcript = "Hi, I'm Dana, thanks for calling, yes, that's right."
    lead = _lead_with(role=("Chief Executive Officer", "yes"))
    verified, rejected = verify_lead(lead, transcript)
    assert "role" in rejected
    assert verified.role.value is None


def test_value_genuinely_derivable_from_its_own_evidence_passes():
    transcript = "Hi, I'm Priya from Acme Corp, interested in pricing."
    lead = _lead_with(name=("Priya", "I'm Priya"))
    verified, rejected = verify_lead(lead, transcript)
    assert "name" not in rejected
    assert verified.name.value == "Priya"


def test_spoken_email_value_derivable_from_spoken_evidence_passes():
    transcript = "You can reach me at john at example dot com any time."
    lead = _lead_with(email=("john@example.com", "john at example dot com"))
    verified, rejected = verify_lead(lead, transcript)
    assert "email" not in rejected
    assert verified.email.value == "john@example.com"


def test_value_supported_by_evidence_direct_unit_checks():
    assert value_supported_by_evidence("Priya", "I'm Priya") is True
    assert value_supported_by_evidence("attacker@evil.example", "my email is") is False
    assert value_supported_by_evidence("CEO", "yes") is False
    assert value_supported_by_evidence("", "anything") is False
    assert value_supported_by_evidence("anything", "") is False


def test_whole_transcript_as_evidence_is_rejected_by_the_length_cap():
    """Verification-review finding 3, reviewer's repro: passing the ENTIRE
    transcript as "evidence" used to let any value whose words appear
    anywhere in it (in any order) through, since the old check was an
    unordered token-set membership test with no bound on evidence length.
    A whole transcript as evidence must now be rejected outright by the
    length cap, regardless of what the value is.
    """
    transcript = (
        "Hi, I'm John Doe calling from a conference booth, not about any "
        "particular deal, just following up generally since our last chat "
        "about networking opportunities at the expo last month. By the "
        "way our new support domain is example.com for tickets. Also, "
        "Mary stopped by earlier asking about enterprise pricing options "
        "for her logistics team deployment next quarter."
    )
    assert len(transcript) > 200
    lead = _lead_with(email=("mary@example.com", transcript))
    verified, rejected = verify_lead(lead, transcript)
    assert "email" in rejected
    assert verified.email.value is None


def test_email_spliced_from_two_unrelated_mentions_within_a_short_span_is_rejected():
    """Reviewer's exact repro: "mary@example.com" is assembled from a
    mention of "Mary" and an unrelated mention of a domain
    ("example.com", actually John's support domain, not Mary's). Even
    though this evidence span is short enough to pass the length cap, the
    two halves are too far apart (in tokens) for this to be the value
    actually being said together, so the ordered-and-proximate check must
    reject it.
    """
    evidence = (
        "Mary asked about enterprise pricing. Separately, our support "
        "domain is example.com for any tickets."
    )
    assert len(evidence) <= 200
    transcript = f"Random preamble. {evidence} Random epilogue."
    lead = _lead_with(email=("mary@example.com", evidence))
    verified, rejected = verify_lead(lead, transcript)
    assert "email" in rejected
    assert verified.email.value is None


def test_phone_number_spliced_from_two_unrelated_numbers_is_rejected():
    """Reviewer's exact repro: a phone number is accepted after splicing
    the first digits of one unrelated number with the last digits of
    another. Both real numbers appear, in order, in a short evidence span,
    so the old unordered/unbounded check (and even a bare ordered-only
    check with no proximity bound) would let the spliced value through;
    the proximity bound must reject it.
    """
    evidence = "My old number was 555-123-4567 and my new office line is 555-999-0000."
    assert len(evidence) <= 200
    spliced_value = "555-123-0000"  # first half of the old number + last half of the new one
    lead = _lead_with(phone=(spliced_value, evidence))
    verified, rejected = verify_lead(lead, evidence)
    assert "phone" in rejected
    assert verified.phone.value is None


def test_genuinely_contiguous_phone_number_still_passes():
    """Control for the proximity-bound fix above: a phone number that is
    genuinely quoted as one contiguous span must still verify."""
    evidence = "You can reach me at 555-123-4567 any time."
    lead = _lead_with(phone=("555-123-4567", evidence))
    verified, rejected = verify_lead(lead, evidence)
    assert "phone" not in rejected
    assert verified.phone.value == "555-123-4567"


def test_evidence_substring_of_transcript_but_value_not_in_evidence_rejected_end_to_end():
    """A record where the evidence span is a genuine transcript substring
    (so the old check alone would have accepted it), but the value is not
    actually said within that span.
    """
    transcript = (
        "This is a voice note about pricing for Globex Corp. "
        "My email is on file already."
    )
    lead = _lead_with(email=("attacker@evil.example", "My email is on file already"))
    verified, rejected = verify_lead(lead, transcript)
    assert "email" in rejected
    assert verified.email.value is None
