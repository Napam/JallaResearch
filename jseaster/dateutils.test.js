const { expect } = require("@jest/globals")
const dateutils = require('./dateutils')

test('Fast count days simple', () => {
  from = new Date(2022, 3, 1)
  to = new Date(2022, 3, 30)
  console.log('dateutils :>> ', dateutils);
  dateutils.fastCountDays(from, to)
})