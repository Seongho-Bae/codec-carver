import re

with open('tests/test_empty_target_validation.py', 'r') as f:
    content = f.read()

content = content.replace('self.assertIn("preview.innerText = \'\';", handler)', 'self.assertIn("preview.innerText = \'This field is required.\';", handler)')
content = content.replace('self.assertIn("this.setCustomValidity(\'\');", handler)', 'self.assertIn("this.setCustomValidity(\'This field is required.\');", handler)')
content = content.replace('self.assertIn("this.removeAttribute(\'aria-invalid\');", handler)', 'self.assertIn("this.setAttribute(\'aria-invalid\', \'true\');", handler)')

with open('tests/test_empty_target_validation.py', 'w') as f:
    f.write(content)
