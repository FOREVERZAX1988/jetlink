#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Write the Sparkle feed for one release: a single item, the DMG on the GitHub
release, with notes from CHANGELOG.md. make-appcast.sh runs it and signs the
result; the feed is attached to the release, where the app finds it at
releases/latest/download/appcast.xml.

One item is enough: Sparkle offers the newest release whatever the installed
one is, and the notes (scripts/changelog.py history) carry the releases before
it, which the app cuts at the one it has.

    python3 macos/scripts/make-appcast.py --app build/Jetlink.app \\
      --archive build/Jetlink-0.9.0-macOS.dmg --signature <sign_update -p> \\
      --url https://github.com/zoompilot/jetlink/releases/download/v0.9.0/Jetlink-0.9.0-macOS.dmg \\
      --release-page https://github.com/zoompilot/jetlink/releases/tag/v0.9.0 \\
      --history https://github.com/zoompilot/jetlink/releases \\
      --notes notes.md --output build/appcast.xml
"""
from __future__ import annotations

import argparse
import plistlib
import sys
from email.utils import formatdate
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


def system_version(value: str) -> str:
  """Sparkle compares a three part version: 15.0 is 15.0.0."""
  parts = value.split('.')
  return '.'.join(parts + ['0'] * (3 - len(parts)))


def cdata(text: str) -> str:
  """Text as CDATA, with any `]]>` in it split across two sections."""
  return '<![CDATA[' + text.replace(']]>', ']]]]><![CDATA[>') + ']]>'


def item_notes(notes: str, release_page: str) -> str:
  """The changelog's notes, or a pointer to the release page without them."""
  return notes.strip() or f'See the [release notes]({release_page}).'


def appcast(*, info: dict, url: str, length: int, signature: str, notes: str, release_page: str, history: str, pub_date: str) -> str:
  short = info['CFBundleShortVersionString']
  minimum = system_version(info.get('LSMinimumSystemVersion', '15.0'))
  return f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Jetlink</title>
    <link>{escape(history)}</link>
    <language>en</language>
    <item>
      <title>Jetlink {escape(short)}</title>
      <link>{escape(release_page)}</link>
      <pubDate>{pub_date}</pubDate>
      <sparkle:version>{escape(info['CFBundleVersion'])}</sparkle:version>
      <sparkle:shortVersionString>{escape(short)}</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>{minimum}</sparkle:minimumSystemVersion>
      <sparkle:hardwareRequirements>arm64</sparkle:hardwareRequirements>
      <sparkle:fullReleaseNotesLink>{escape(history)}</sparkle:fullReleaseNotesLink>
      <description sparkle:format="markdown">{cdata(item_notes(notes, release_page))}</description>
      <enclosure url={quoteattr(url)} length="{length}" type="application/octet-stream" sparkle:edSignature={quoteattr(signature)}/>
    </item>
  </channel>
</rss>
"""


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--app', type=Path, required=True, help='the app inside the archive, for its versions')
  parser.add_argument('--archive', type=Path, required=True, help='the DMG the feed offers')
  parser.add_argument('--signature', required=True, help="the archive's EdDSA signature, from sign_update -p")
  parser.add_argument('--url', required=True, help='where the archive is downloaded from')
  parser.add_argument('--release-page', required=True, help="the release's page, for the item's link")
  parser.add_argument('--history', required=True, help='every release, for Version History')
  parser.add_argument('--notes', type=Path, required=True, help='markdown from scripts/changelog.py history; may be empty')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(argv)

  with open(args.app / 'Contents' / 'Info.plist', 'rb') as f:
    info = plistlib.load(f)
  args.output.write_text(appcast(
    info=info,
    url=args.url,
    length=args.archive.stat().st_size,
    signature=args.signature.strip(),
    notes=args.notes.read_text(encoding='utf-8'),
    release_page=args.release_page,
    history=args.history,
    pub_date=formatdate(usegmt=True),
  ), encoding='utf-8')
  return 0


if __name__ == '__main__':
  sys.exit(main())
