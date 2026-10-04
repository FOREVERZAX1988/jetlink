"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Release notes from CHANGELOG.md (scripts/changelog.py) and the Mac app's
update feed made from them (macos/scripts/make-appcast.py).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from tests import load_script

ROOT = Path(__file__).resolve().parents[1]
changelog = load_script(ROOT / 'scripts' / 'changelog.py')
make_appcast = load_script(ROOT / 'macos' / 'scripts' / 'make-appcast.py')

SPARKLE = '{http://www.andymatuschak.org/xml-namespaces/sparkle}'

SAMPLE = """Jetlink v0.9.0
==============
**Driving**
* One

* Two

Jetlink v0.8.1
==============
* Older


Jetlink v0.8.0
==============
* Oldest
"""


def test_sections_newest_first_without_the_underline():
  assert changelog.sections(SAMPLE) == [
    ('v0.9.0', '**Driving**\n* One\n\n* Two'),
    ('v0.8.1', '* Older'),
    ('v0.8.0', '* Oldest'),
  ]


def test_notes_are_one_section_and_empty_for_an_unknown_tag():
  assert changelog.notes(SAMPLE, 'v0.8.1') == '* Older'
  assert changelog.notes(SAMPLE, 'v9.9.9') == ''


def test_history_puts_each_release_under_its_heading():
  assert changelog.history(SAMPLE, 'v0.9.0', 2) == '## Jetlink v0.9.0\n\n**Driving**\n* One\n\n* Two\n\n## Jetlink v0.8.1\n\n* Older'
  assert changelog.history(SAMPLE, 'v0.8.1', 10) == '## Jetlink v0.8.1\n\n* Older\n\n## Jetlink v0.8.0\n\n* Oldest'
  assert changelog.history(SAMPLE, 'v9.9.9', 10) == ''


def test_the_real_changelog_parses():
  text = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
  releases = changelog.sections(text)
  assert len(releases) > 5
  assert all(tag.startswith('v') and body for tag, body in releases)
  assert len({tag for tag, _ in releases}) == len(releases)


def feed(notes='## Jetlink v0.9.0\n\n* One'):
  xml = make_appcast.appcast(
    info={'CFBundleShortVersionString': '0.9.0', 'CFBundleVersion': '961', 'LSMinimumSystemVersion': '15.0'},
    url='https://github.com/zoompilot/jetlink/releases/download/v0.9.0/Jetlink-0.9.0-macOS.dmg',
    length=12345,
    signature='c2lnbmF0dXJl==',
    notes=notes,
    release_page='https://github.com/zoompilot/jetlink/releases/tag/v0.9.0',
    history='https://github.com/zoompilot/jetlink/releases',
    pub_date='Sat, 03 Oct 2026 20:00:00 GMT',
  )
  return ET.fromstring(xml).find('channel/item')


def test_the_feed_item_carries_what_sparkle_compares():
  item = feed()
  assert item.findtext('title') == 'Jetlink 0.9.0'
  assert item.findtext(f'{SPARKLE}version') == '961'
  assert item.findtext(f'{SPARKLE}shortVersionString') == '0.9.0'
  assert item.findtext(f'{SPARKLE}minimumSystemVersion') == '15.0.0'
  assert item.findtext(f'{SPARKLE}hardwareRequirements') == 'arm64'
  enclosure = item.find('enclosure')
  assert enclosure.get('url').endswith('/v0.9.0/Jetlink-0.9.0-macOS.dmg')
  assert enclosure.get('length') == '12345'
  assert enclosure.get(f'{SPARKLE}edSignature') == 'c2lnbmF0dXJl=='


def test_the_notes_are_markdown_and_survive_any_text():
  item = feed()
  assert item.find('description').get(f'{SPARKLE}format') == 'markdown'
  assert item.findtext('description') == '## Jetlink v0.9.0\n\n* One'
  tricky = '* a ]]> b & <c>'
  assert feed(tricky).findtext('description') == tricky


def test_a_release_without_notes_links_its_page():
  assert feed('').findtext('description') == 'See the [release notes](https://github.com/zoompilot/jetlink/releases/tag/v0.9.0).'
