"""
The MIT License (MIT)

Copyright (c) 2025-present Seizh7

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from capitolwatch.parsing.cleaner import clean_html_string

def test_clean_html_realistic_senate_report_anonymized():
    html = """
    <html>
      <head><title>eFD: Annual Report for 2023 - Dupont, Pierre X</title></head>
      <body>
        <h2 class="filedReport">
            The Honorable
            Pierre
            X
            Dupont
            (Dupont, Pierre X.)
        </h2>
        <section class="card mb-2">
            <div class="card-body">
                <h3 class="h4">Part 3. Assets</h3>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Asset</th>
                                <th>Asset Type</th>
                                <th>Owner</th>
                                <th>Value</th>
                                <th>Income Type</th>
                                <th>Income</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Bank A</strong><div class="muted noWrap">(City, ST)</div></td>
                                <td>Bank Deposit</td>
                                <td>Joint</td>
                                <td>$10,001 - $50,000</td>
                                <td>Interest, </td>
                                <td>None (or less than $201)</td>
                            </tr>
                            <tr>
                                <td><strong>Main Residence</strong></td>
                                <td>Real Estate</td>
                                <td>Joint</td>
                                <td>$500,001 - $1,000,000</td>
                                <td>Rent/Royalties, </td>
                                <td>$5,001 - $15,000</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
      </body>
    </html>
    """
    cleaned = clean_html_string(html)
    # Check for essential information in the cleaned text (anonymized)
    assert "Pierre" in cleaned
    assert "Dupont" in cleaned
    assert "Bank A" in cleaned
    assert "Main Residence" in cleaned
    assert "Bank Deposit" in cleaned
    assert "$500,001 - $1,000,000" in cleaned
    assert "Interest" in cleaned
    assert "Rent/Royalties" in cleaned

    # Negative controls: removed HTML tags and classes
    assert "<div" not in cleaned
    assert "</td>" not in cleaned
    assert "noWrap" not in cleaned
