"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT license.
See the LICENSE file in the root directory for more details.

The source selection the registry fetches through: which URL shape a CDN
takes, which one a proxy takes, and how the candidates are ordered.
"""
from __future__ import annotations

from jetlink.registry import sources
from jetlink.registry.sources import (clear_cache, jsdelivr_candidates, probe_sorted,
                                      proxy_candidates, raw_to_hf_mirror, raw_to_jsdelivr)

RAW = 'https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_v25.json'
LEGACY = 'https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/gh-pages/docs/driving_models_v25.json'
HF = 'https://huggingface.co/api/models/commaai/openpilot_driving_models/tree/main'


class TestShapes:
  """Two shapes, and mixing them fails."""

  def test_the_cdn_takes_a_path(self):
    assert raw_to_jsdelivr(RAW) == \
      'https://cdn.jsdelivr.net/gh/sunnypilot/sunnypilot-models@gh-pages/docs/driving_models_v25.json'

  def test_the_cdn_takes_the_legacy_path_too(self):
    assert raw_to_jsdelivr(LEGACY) == \
      'https://cdn.jsdelivr.net/gh/sunnypilot/sunnypilot-models@gh-pages/docs/driving_models_v25.json'

  def test_the_cdn_takes_a_host_of_its_own(self):
    assert raw_to_jsdelivr(RAW, 'https://fastly.jsdelivr.net').startswith('https://fastly.jsdelivr.net/gh/')

  def test_the_cdn_does_not_take_another_host(self):
    assert raw_to_jsdelivr('https://example.com/a/b/c/d') is None
    assert raw_to_jsdelivr('https://raw.githubusercontent.com/owner/repo') is None

  def test_a_proxy_takes_a_prefix(self):
    assert proxy_candidates(RAW)[0] == 'https://gh-proxy.com/' + RAW

  def test_a_mirror_takes_a_prefix(self):
    assert raw_to_hf_mirror(HF, 'https://hf-mirror.net') == \
      'https://hf-mirror.net/api/models/commaai/openpilot_driving_models/tree/main'

  def test_a_mirror_does_not_take_another_host(self):
    assert raw_to_hf_mirror('https://example.com/x', 'https://hf-mirror.net') is None

  def test_the_cdn_family_is_asked_before_the_proxies(self):
    """The CDN is one service with a stable path form; the proxies are
    third-party services that come and go. The order decides what gets
    measured when there are more candidates than the probe budget allows."""
    urls = jsdelivr_candidates(RAW) + proxy_candidates(RAW)
    assert urls.index(jsdelivr_candidates(RAW)[0]) < urls.index(proxy_candidates(RAW)[0])


class TestProbeSorted:
  def setup_method(self):
    clear_cache()

  def test_a_single_candidate_is_left_alone(self):
    assert probe_sorted([RAW]) == [RAW]

  def test_the_fastest_comes_first(self, monkeypatch):
    delays = {'https://slow.example/x': 900, 'https://fast.example/x': 100, 'https://mid.example/x': 500}
    monkeypatch.setattr(sources, '_measure', lambda url, timeout=0, opener=None: (True, delays[url]))
    urls = list(delays)
    assert probe_sorted(urls) == ['https://fast.example/x', 'https://mid.example/x', 'https://slow.example/x']

  def test_a_dead_source_goes_last(self, monkeypatch):
    delays = {'https://dead.example/x': 0, 'https://live.example/x': 800}
    def measure(url, timeout=0, opener=None):
      ok = url != 'https://dead.example/x'
      return ok, delays[url]
    monkeypatch.setattr(sources, '_measure', measure)
    urls = list(delays)
    assert probe_sorted(urls) == ['https://live.example/x', 'https://dead.example/x']

  def test_nothing_answering_keeps_the_order(self, monkeypatch):
    monkeypatch.setattr(sources, '_measure', lambda url, timeout=0, opener=None: (False, 0))
    urls = ['https://a.example/x', 'https://b.example/x']
    assert probe_sorted(urls) == urls

  def test_the_first_candidate_can_be_kept_where_it_is(self, monkeypatch):
    """The official source stays first: a 404 from it means the file is not
    there, which the catalog's version probe relies on."""
    monkeypatch.setattr(sources, '_measure', lambda url, timeout=0, opener=None: (True, 900))
    urls = ['https://official.example/x', 'https://cdn.example/x']
    assert probe_sorted(urls, keep_first=True)[0] == 'https://official.example/x'

  def test_only_a_few_hosts_are_measured(self, monkeypatch):
    calls = []
    def measure(url, timeout=0, opener=None):
      calls.append(url)
      return True, 100
    monkeypatch.setattr(sources, '_measure', measure)
    urls = [f'https://s{i}.example/x' for i in range(10)]
    out = probe_sorted(urls, max_probe=3)
    assert len(calls) == 3
    assert out[:3] == urls[:3], 'the measured ones come first'
    assert out[3:] == urls[3:], 'the unmeasured keep their order'

  def test_a_host_is_measured_once(self, monkeypatch):
    calls = []
    def measure(url, timeout=0, opener=None):
      calls.append(url)
      return True, 100
    monkeypatch.setattr(sources, '_measure', measure)
    probe_sorted(['https://same.example/a', 'https://same.example/b'])
    assert len(calls) == 1
